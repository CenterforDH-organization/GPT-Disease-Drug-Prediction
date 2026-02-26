import scipy.stats
import scipy
import warnings
import torch
# Suppress sklearn warnings about classes not in y_true
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')
from model_loss_update import ModernDelphi, ModernDelphiConfig, CompositeDelphi, CompositeDelphiConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from utils import get_batch, get_p2i, get_batch_composite, get_p2i_composite
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    top_k_accuracy_score, 
    mean_absolute_error, 
    mean_squared_error,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    r2_score,
    confusion_matrix,
    roc_auc_score,
)


def auc(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def get_common_diseases(labels_df, filter_min_total=100):
    if 'count' in labels_df.columns:
        labels_df_filtered = labels_df[labels_df['count'] > filter_min_total]
    else:
        labels_df_filtered = labels_df[labels_df['index'] > 20]
    raw_indices = labels_df_filtered['index'].tolist()
    shifted_tokens = [idx + 1 for idx in raw_indices]
    return shifted_tokens


def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=1000):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")
    if not torch.is_tensor(case):
        case = torch.tensor(case, device="cuda", dtype=torch.float32)
    else:
        case = case.to("cuda", dtype=torch.float32)
    if not torch.is_tensor(control):
        control = torch.tensor(control, device="cuda", dtype=torch.float32)
    else:
        control = control.to("cuda", dtype=torch.float32)
    n_case = case.size(0)
    n_control = control.size(0)
    total = n_case + n_control
    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_control = torch.randint(0, n_control, (n_bootstrap, n_control), device="cuda")
    boot_case = case[boot_idx_case]
    boot_control = control[boot_idx_control]
    combined = torch.cat([boot_case, boot_control], dim=1)
    mask = torch.zeros((n_bootstrap, total), dtype=torch.bool, device="cuda")
    mask[:, :n_case] = True
    ranks = combined.argsort(dim=1).argsort(dim=1)
    case_ranks_sum = torch.sum(ranks.float() * mask.float(), dim=1)
    min_case_rank_sum = n_case * (n_case - 1) / 2.0
    U = case_ranks_sum - min_case_rank_sum
    aucs = U / (n_case * n_control)
    return aucs.cpu().tolist()


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        if m > 1 and v01.shape[1] > 1:
            sx = np.cov(v01)
            if np.any(np.isnan(sx)) or np.any(np.isinf(sx)):
                sx = np.zeros_like(sx)
        else:
            sx = np.zeros((k, k), dtype=np.float32)
        if n > 1 and v10.shape[1] > 1:
            sy = np.cov(v10)
            if np.any(np.isnan(sy)) or np.any(np.isinf(sy)):
                sy = np.zeros_like(sy)
        else:
            sy = np.zeros((k, k), dtype=np.float32)
    if m > 0 and n > 0:
        delongcov = sx / m + sy / n
        delongcov = np.nan_to_num(delongcov, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        delongcov = np.zeros((k, k), dtype=np.float32)
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def get_auc_delong_var(healthy_scores, diseased_scores):
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1
    if isinstance(delongcov, np.ndarray):
        delongcov = delongcov.item() if delongcov.size == 1 else delongcov[0, 0]
    return aucs[0], delongcov


def get_calibration_auc(j, k, d, p, diseases_chunk, offset=365.25, age_groups=range(45, 80, 5), precomputed_idx=None, n_bootstrap=1, use_delong=False):
    age_step = age_groups[1] - age_groups[0]
    wk = np.where(d[2] == k)
    n_cases = len(wk[0])
    patient_has_disease = (d[2] == k).any(axis=1)
    wc = np.where((d[2] != k) & (~patient_has_disease[:, None]))
    n_controls = len(wc[0])
    if n_cases < 2 or n_controls == 0:
        out = []
        reason = "insufficient_cases" if n_cases < 2 else "no_controls"
        for aa in age_groups:
            out_item = {"token": k, "auc": np.nan, "age": aa, "n_healthy": n_controls if n_controls > 0 else 0, "n_diseased": n_cases, "status": reason}
            if use_delong:
                out_item["auc_delong"] = np.nan
                out_item["auc_variance_delong"] = np.nan
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = 0
            out.append(out_item)
        return out
    wall = (np.concatenate([wk[0], wc[0]]), np.concatenate([wk[1], wc[1]]))
    if precomputed_idx is None:
        pred_idx = (d[1][wall[0]] <= d[3][wall].reshape(-1, 1) - offset).sum(1) - 1
    else:
        pred_idx = precomputed_idx[wall]
    z = d[1][(wall[0], pred_idx)]
    z = z[pred_idx != -1]
    zk = d[3][wall]
    zk = zk[pred_idx != -1]
    x = p[..., j][(wall[0], pred_idx)]
    x = x[pred_idx != -1]
    wk = (wk[0][pred_idx[: len(wk[0])] != -1], wk[1][pred_idx[: len(wk[0])] != -1])
    p_idx = wall[0][pred_idx != -1]
    out = []
    for i, aa in enumerate(age_groups):
        a = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + age_step)
        selected_groups = p_idx[a]
        perm = np.random.permutation(len(selected_groups))
        _, indices = np.unique(selected_groups[perm], return_index=True)
        indices = perm[indices]
        selected = np.zeros(np.sum(a), dtype=bool)
        selected[indices] = True
        a[a] = selected
        control = x[len(wk[0]) :][a[len(wk[0]) :]]
        case = x[: len(wk[0])][a[: len(wk[0])]]
        if len(control) == 0 or len(case) == 0:
            continue
        if use_delong:
            auc_value_delong, auc_variance_delong = get_auc_delong_var(control, case)
            if isinstance(auc_variance_delong, np.ndarray):
                auc_variance_delong = auc_variance_delong.item() if auc_variance_delong.size == 1 else float(auc_variance_delong[0, 0])
            else:
                auc_variance_delong = float(auc_variance_delong)
            auc_delong_dict = {"auc_delong": float(auc_value_delong), "auc_variance_delong": auc_variance_delong}
        else:
            auc_delong_dict = {}
        if n_bootstrap > 1:
            aucs_bootstrapped = optimized_bootstrapped_auc_gpu(case, control, n_bootstrap)
        for bootstrap_idx in range(n_bootstrap):
            if n_bootstrap == 1:
                if use_delong:
                    y = auc_value_delong
                else:
                    y = auc(case, control)
            else:
                y = aucs_bootstrapped[bootstrap_idx]
            out_item = {"token": k, "auc": y, "age": aa, "n_healthy": len(control), "n_diseased": len(case), "status": "ok"}
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = bootstrap_idx
            out.append(out_item | auc_delong_dict)
    return out


def _compute_shift_auc(shift_target, shift_probs, unique_classes=np.array([1, 2, 3])):
    """Compute SHIFT AUC (macro/weighted OVR + per-class)."""
    results = {}
    try:
        # Filter probs to classes 1, 2, 3 columns
        probs_filtered = shift_probs[:, unique_classes]
        probs_filtered = probs_filtered / (probs_filtered.sum(axis=1, keepdims=True) + 1e-8)

        present_classes = np.unique(shift_target)
        if len(present_classes) >= 2:
            target_mapped = shift_target - 1  # 1,2,3 -> 0,1,2
            results['auc_macro'] = float(roc_auc_score(
                target_mapped, probs_filtered, multi_class='ovr', average='macro'))
            results['auc_weighted'] = float(roc_auc_score(
                target_mapped, probs_filtered, multi_class='ovr', average='weighted'))

            per_class = {}
            class_name_map = {1: 'Decrease', 2: 'Maintain', 3: 'Increase'}
            for ci, cls in enumerate(unique_classes):
                bt = (shift_target == cls).astype(int)
                if bt.sum() > 0 and (1 - bt).sum() > 0:
                    try:
                        per_class[int(cls)] = float(roc_auc_score(bt, probs_filtered[:, ci]))
                    except Exception:
                        per_class[int(cls)] = np.nan
                else:
                    per_class[int(cls)] = np.nan
            results['per_class_auc'] = per_class
        else:
            results['auc_macro'] = np.nan
            results['auc_weighted'] = np.nan
    except Exception as e:
        results['auc_macro'] = np.nan
        results['auc_weighted'] = np.nan
        print(f"  [WARN] SHIFT AUC computation failed: {e}")
    return results


def _compute_change_auc(change_targets, change_probs):
    """Compute binary change detection AUC."""
    try:
        if len(np.unique(change_targets)) == 2:
            return float(roc_auc_score(change_targets, change_probs))
    except Exception:
        pass
    return np.nan


def evaluate_composite_fields(model, d100k, batch_size=64, device="mps"):
    """
    Evaluate SHIFT, CHANGE, TOTAL predictions for CompositeDelphi model.
    """
    model.eval()
    model.to(device)
    all_predictions = {'shift': [], 'total': []}
    all_targets = {'shift': [], 'total': []}
    all_shift_probs = []  # For SHIFT AUC
    all_predictions_pos = {'total': []}
    all_targets_pos = {'total': []}

    # Change detection
    all_change_preds = []
    all_change_targets = []
    all_change_probs = []
    all_change_preds_drug = []
    all_change_targets_drug = []
    all_change_probs_drug = []

    # Drug-conditioned
    all_predictions_shift_drug_cond = []
    all_predictions_total_drug_cond = []
    all_targets_shift_drug_cond = []
    all_targets_total_drug_cond = []
    all_shift_probs_drug_cond = []  # For drug-conditioned SHIFT AUC
    use_drug_conditioning = getattr(model.config, 'use_drug_conditioning', False)

    has_change_head = False

    x_data, x_shift, x_total, x_ages = d100k[0], d100k[1], d100k[2], d100k[3]
    y_data, y_shift, y_total, y_ages = d100k[4], d100k[5], d100k[6], d100k[7]

    # Determine maintain index for change detection
    shift_maintain_idx = getattr(model.config, 'shift_maintain_idx', 2)

    num_batches = (x_data.shape[0] + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating composite fields"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, x_data.shape[0])

            batch_x_data = x_data[start_idx:end_idx].to(device)
            batch_x_shift = x_shift[start_idx:end_idx].to(device)
            batch_x_total = x_total[start_idx:end_idx].to(device)
            batch_x_ages = x_ages[start_idx:end_idx].to(device)
            batch_y_data = y_data[start_idx:end_idx].to(device)
            batch_y_shift = y_shift[start_idx:end_idx].to(device)
            batch_y_total = y_total[start_idx:end_idx].to(device)

            outputs = model(
                batch_x_data, batch_x_shift, batch_x_total, batch_x_ages,
                targets_data=batch_y_data, targets_shift=batch_y_shift,
                targets_total=batch_y_total,
                targets_age=y_ages[start_idx:end_idx].to(device)
            )[0]

            shift_logits = outputs['shift']
            total_pred = outputs['total']
            shift_pred = torch.argmax(shift_logits, dim=-1)
            shift_probs = torch.softmax(shift_logits, dim=-1)  # (B, T, C)

            shift_mask = (batch_y_shift != -1) & (batch_y_shift > 0)
            total_mask = (batch_y_total != -1) & (batch_y_total >= 0)

            drug_token_min = getattr(model.config, 'drug_token_min', 1279)
            drug_token_max = getattr(model.config, 'drug_token_max', 1289)
            drug_token_mask = (batch_y_data >= drug_token_min) & (batch_y_data <= drug_token_max)

            # ---- CHANGE detection ----
            if 'change' in outputs:
                has_change_head = True
                change_logits = outputs['change']  # (B, T, 2)
                change_pred = torch.argmax(change_logits, dim=-1)
                change_prob = torch.softmax(change_logits, dim=-1)[:, :, 1]  # P(changed)
                change_target = (batch_y_shift != shift_maintain_idx).long()

                if shift_mask.any():
                    all_change_preds.append(change_pred[shift_mask].cpu().numpy())
                    all_change_targets.append(change_target[shift_mask].cpu().numpy())
                    all_change_probs.append(change_prob[shift_mask].cpu().numpy())

                    drug_change_mask = shift_mask & drug_token_mask
                    if drug_change_mask.any():
                        all_change_preds_drug.append(change_pred[drug_change_mask].cpu().numpy())
                        all_change_targets_drug.append(change_target[drug_change_mask].cpu().numpy())
                        all_change_probs_drug.append(change_prob[drug_change_mask].cpu().numpy())

            # ---- SHIFT ----
            if shift_mask.any():
                all_predictions['shift'].append(shift_pred[shift_mask].cpu().numpy())
                all_targets['shift'].append(batch_y_shift[shift_mask].cpu().numpy())
                all_shift_probs.append(shift_probs[shift_mask].cpu().numpy())

                if use_drug_conditioning and 'shift_drug_cond' in outputs:
                    shift_drug_logits = outputs['shift_drug_cond']
                    shift_drug_pred = torch.argmax(shift_drug_logits, dim=-1)
                    shift_drug_probs = torch.softmax(shift_drug_logits, dim=-1)
                    drug_shift_mask = shift_mask & drug_token_mask
                    if drug_shift_mask.any():
                        all_predictions_shift_drug_cond.append(shift_drug_pred[drug_shift_mask].cpu().numpy())
                        all_targets_shift_drug_cond.append(batch_y_shift[drug_shift_mask].cpu().numpy())
                        all_shift_probs_drug_cond.append(shift_drug_probs[drug_shift_mask].cpu().numpy())

            # ---- TOTAL ----
            if total_mask.any():
                tp = total_pred[total_mask]
                tt = batch_y_total[total_mask].float()
                all_predictions['total'].append(torch.clamp(tp, min=0.0).cpu().numpy())
                all_targets['total'].append(tt.cpu().numpy())

                if use_drug_conditioning and 'total_drug_cond' in outputs:
                    drug_total_mask = total_mask & drug_token_mask
                    if drug_total_mask.any():
                        tp_drug = outputs['total_drug_cond'][drug_total_mask]
                        tp_drug = torch.clamp(tp_drug, min=0.0)
                        all_predictions_total_drug_cond.append(tp_drug.cpu().numpy())
                        all_targets_total_drug_cond.append(batch_y_total[drug_total_mask].float().cpu().numpy())

                total_pos = total_mask & (batch_y_total > 0)
                if total_pos.any():
                    tp_pos = total_pred[total_pos]
                    tt_pos = batch_y_total[total_pos].float()
                    all_predictions_pos['total'].append(torch.clamp(tp_pos, min=0.0).cpu().numpy())
                    all_targets_pos['total'].append(tt_pos.cpu().numpy())

    # ---- Concatenate ----
    for field in ['shift', 'total']:
        if len(all_predictions[field]) > 0:
            all_predictions[field] = np.concatenate(all_predictions[field])
            all_targets[field] = np.concatenate(all_targets[field])
        else:
            all_predictions[field] = np.array([])
            all_targets[field] = np.array([])

    all_shift_probs = np.concatenate(all_shift_probs) if len(all_shift_probs) > 0 else np.array([])

    for field in ['total']:
        if len(all_predictions_pos[field]) > 0:
            all_predictions_pos[field] = np.concatenate(all_predictions_pos[field])
            all_targets_pos[field] = np.concatenate(all_targets_pos[field])
        else:
            all_predictions_pos[field] = np.array([])
            all_targets_pos[field] = np.array([])

    all_predictions_shift_drug_cond = np.concatenate(all_predictions_shift_drug_cond) if len(all_predictions_shift_drug_cond) > 0 else np.array([])
    all_targets_shift_drug_cond = np.concatenate(all_targets_shift_drug_cond) if len(all_targets_shift_drug_cond) > 0 else np.array([])
    all_shift_probs_drug_cond = np.concatenate(all_shift_probs_drug_cond) if len(all_shift_probs_drug_cond) > 0 else np.array([])
    all_predictions_total_drug_cond = np.concatenate(all_predictions_total_drug_cond) if len(all_predictions_total_drug_cond) > 0 else np.array([])
    all_targets_total_drug_cond = np.concatenate(all_targets_total_drug_cond) if len(all_targets_total_drug_cond) > 0 else np.array([])

    all_change_preds = np.concatenate(all_change_preds) if len(all_change_preds) > 0 else np.array([])
    all_change_targets = np.concatenate(all_change_targets) if len(all_change_targets) > 0 else np.array([])
    all_change_probs = np.concatenate(all_change_probs) if len(all_change_probs) > 0 else np.array([])
    all_change_preds_drug = np.concatenate(all_change_preds_drug) if len(all_change_preds_drug) > 0 else np.array([])
    all_change_targets_drug = np.concatenate(all_change_targets_drug) if len(all_change_targets_drug) > 0 else np.array([])
    all_change_probs_drug = np.concatenate(all_change_probs_drug) if len(all_change_probs_drug) > 0 else np.array([])

    # ============================================================
    # Calculate metrics
    # ============================================================
    results = {}

    # ---- CHANGE: Binary classification ----
    if has_change_head and len(all_change_targets) > 0:
        results['change_accuracy'] = accuracy_score(all_change_targets, all_change_preds)
        results['change_balanced_accuracy'] = balanced_accuracy_score(all_change_targets, all_change_preds)
        results['change_f1_binary'] = f1_score(all_change_targets, all_change_preds, average='binary', pos_label=1, zero_division=0)
        results['change_precision'] = precision_score(all_change_targets, all_change_preds, average='binary', pos_label=1, zero_division=0)
        results['change_recall'] = recall_score(all_change_targets, all_change_preds, average='binary', pos_label=1, zero_division=0)
        results['change_support'] = int(len(all_change_targets))
        results['change_n_changed'] = int((all_change_targets == 1).sum())
        results['change_n_maintain'] = int((all_change_targets == 0).sum())
        cm_ch = confusion_matrix(all_change_targets, all_change_preds, labels=[0, 1])
        results['change_confusion_matrix'] = cm_ch.tolist()
        results['change_auc'] = _compute_change_auc(all_change_targets, all_change_probs)

        # Drug-conditioned change
        if len(all_change_preds_drug) > 0:
            results['change_accuracy_drug'] = accuracy_score(all_change_targets_drug, all_change_preds_drug)
            results['change_balanced_accuracy_drug'] = balanced_accuracy_score(all_change_targets_drug, all_change_preds_drug)
            results['change_f1_binary_drug'] = f1_score(all_change_targets_drug, all_change_preds_drug, average='binary', pos_label=1, zero_division=0)
            results['change_precision_drug'] = precision_score(all_change_targets_drug, all_change_preds_drug, average='binary', pos_label=1, zero_division=0)
            results['change_recall_drug'] = recall_score(all_change_targets_drug, all_change_preds_drug, average='binary', pos_label=1, zero_division=0)
            results['change_support_drug'] = int(len(all_change_targets_drug))
            results['change_n_changed_drug'] = int((all_change_targets_drug == 1).sum())
            results['change_n_maintain_drug'] = int((all_change_targets_drug == 0).sum())
            cm_ch_drug = confusion_matrix(all_change_targets_drug, all_change_preds_drug, labels=[0, 1])
            results['change_confusion_matrix_drug'] = cm_ch_drug.tolist()
            results['change_auc_drug'] = _compute_change_auc(all_change_targets_drug, all_change_probs_drug)

    # ---- SHIFT: Classification ----
    if len(all_targets['shift']) > 0:
        shift_pred = all_predictions['shift']
        shift_target = all_targets['shift']
        unique_classes = np.array([1, 2, 3])

        results['shift_accuracy'] = accuracy_score(shift_target, shift_pred)
        results['shift_balanced_accuracy'] = balanced_accuracy_score(shift_target, shift_pred)
        results['shift_f1_macro'] = f1_score(shift_target, shift_pred, average='macro', zero_division=0)
        results['shift_f1_micro'] = f1_score(shift_target, shift_pred, average='micro', zero_division=0)
        results['shift_f1_weighted'] = f1_score(shift_target, shift_pred, average='weighted', zero_division=0)
        results['shift_precision_macro'] = precision_score(shift_target, shift_pred, average='macro', zero_division=0)
        results['shift_recall_macro'] = recall_score(shift_target, shift_pred, average='macro', zero_division=0)
        results['shift_support'] = int(len(shift_target))

        # SHIFT AUC
        if len(all_shift_probs) > 0:
            shift_auc = _compute_shift_auc(shift_target, all_shift_probs, unique_classes)
            results['shift_auc_macro'] = shift_auc.get('auc_macro', np.nan)
            results['shift_auc_weighted'] = shift_auc.get('auc_weighted', np.nan)
            if 'per_class_auc' in shift_auc:
                results['shift_per_class_auc'] = shift_auc['per_class_auc']

        cm = confusion_matrix(shift_target, shift_pred, labels=unique_classes)
        results['shift_confusion_matrix'] = cm.tolist()
        results['shift_confusion_matrix_classes'] = unique_classes.tolist()

        per_class_metrics = {}
        for i, cls in enumerate(unique_classes):
            tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_metrics[int(cls)] = {'precision': float(prec), 'recall': float(rec), 'f1': float(f1c), 'support': int(tp + fn)}
        results['shift_per_class_metrics'] = per_class_metrics

        # Drug-conditioned SHIFT
        if len(all_predictions_shift_drug_cond) > 0:
            shift_pred_drug = all_predictions_shift_drug_cond
            shift_target_drug = all_targets_shift_drug_cond
            class_labels = np.array([1, 2, 3])

            results['shift_accuracy_drug_cond'] = accuracy_score(shift_target_drug, shift_pred_drug)
            results['shift_balanced_accuracy_drug_cond'] = balanced_accuracy_score(shift_target_drug, shift_pred_drug)
            results['shift_f1_macro_drug_cond'] = f1_score(shift_target_drug, shift_pred_drug, average='macro', zero_division=0)
            results['shift_f1_micro_drug_cond'] = f1_score(shift_target_drug, shift_pred_drug, average='micro', zero_division=0)
            results['shift_f1_weighted_drug_cond'] = f1_score(shift_target_drug, shift_pred_drug, average='weighted', zero_division=0)
            results['shift_precision_macro_drug_cond'] = precision_score(shift_target_drug, shift_pred_drug, average='macro', zero_division=0)
            results['shift_recall_macro_drug_cond'] = recall_score(shift_target_drug, shift_pred_drug, average='macro', zero_division=0)
            results['shift_support_drug_cond'] = int(len(shift_target_drug))

            # Drug-conditioned SHIFT AUC
            if len(all_shift_probs_drug_cond) > 0:
                shift_auc_drug = _compute_shift_auc(shift_target_drug, all_shift_probs_drug_cond, class_labels)
                results['shift_auc_macro_drug_cond'] = shift_auc_drug.get('auc_macro', np.nan)
                results['shift_auc_weighted_drug_cond'] = shift_auc_drug.get('auc_weighted', np.nan)
                if 'per_class_auc' in shift_auc_drug:
                    results['shift_per_class_auc_drug_cond'] = shift_auc_drug['per_class_auc']

            cm_drug = confusion_matrix(shift_target_drug, shift_pred_drug, labels=class_labels)
            results['shift_confusion_matrix_drug_cond'] = cm_drug.tolist()
            results['shift_confusion_matrix_drug_cond_classes'] = class_labels.tolist()

            per_class_metrics_drug = {}
            for i, cls in enumerate(class_labels):
                tp = cm_drug[i, i]; fp = cm_drug[:, i].sum() - tp; fn = cm_drug[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class_metrics_drug[int(cls)] = {'precision': float(prec), 'recall': float(rec), 'f1': float(f1c), 'support': int(tp + fn)}
            results['shift_per_class_metrics_drug_cond'] = per_class_metrics_drug
            results['shift_drug_cond_note'] = 'Metrics computed only for drug tokens (1279-1289)'

    # ---- TOTAL: Regression ----
    for field in ['total']:
        if len(all_targets[field]) == 0:
            continue
        pred = all_predictions[field]
        target = all_targets[field]
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        median_ae = np.median(np.abs(target - pred))
        try:
            r2 = r2_score(target, pred)
        except:
            r2 = np.nan
        results[f'{field}_mae'] = mae
        results[f'{field}_rmse'] = rmse
        results[f'{field}_median_ae'] = median_ae
        results[f'{field}_r2'] = r2
        results[f'{field}_mean_target'] = np.mean(target)
        results[f'{field}_mean_pred'] = np.mean(pred)
        results[f'{field}_std_target'] = np.std(target)
        results[f'{field}_std_pred'] = np.std(pred)

        if len(all_targets_pos[field]) > 0:
            pred_pos = all_predictions_pos[field]
            target_pos = all_targets_pos[field]
            results[f'{field}_mae_pos'] = mean_absolute_error(target_pos, pred_pos)
            results[f'{field}_rmse_pos'] = float(np.sqrt(mean_squared_error(target_pos, pred_pos)))
            results[f'{field}_median_ae_pos'] = float(np.median(np.abs(target_pos - pred_pos)))
            try:
                results[f'{field}_r2_pos'] = r2_score(target_pos, pred_pos)
            except:
                results[f'{field}_r2_pos'] = np.nan
            results[f'{field}_support_pos'] = int(len(target_pos))

    # Drug-conditioned TOTAL
    if len(all_predictions_total_drug_cond) > 0:
        pred_drug = all_predictions_total_drug_cond
        tgt_drug = all_targets_total_drug_cond
        results['total_mae_drug_cond'] = mean_absolute_error(tgt_drug, pred_drug)
        results['total_rmse_drug_cond'] = float(np.sqrt(mean_squared_error(tgt_drug, pred_drug)))
        results['total_median_ae_drug_cond'] = float(np.median(np.abs(tgt_drug - pred_drug)))
        try:
            results['total_r2_drug_cond'] = r2_score(tgt_drug, pred_drug)
        except:
            results['total_r2_drug_cond'] = np.nan
        results['total_mean_target_drug_cond'] = float(np.mean(tgt_drug))
        results['total_mean_pred_drug_cond'] = float(np.mean(pred_drug))
        results['total_support_drug_cond'] = int(len(tgt_drug))
        results['total_drug_cond_note'] = 'Metrics computed only for drug tokens (1279-1289)'

    return results


def _print_composite_metrics(composite_metrics):
    """Print all composite field evaluation results."""
    print("\nComposite Field Evaluation Results:")
    print("=" * 60)

    # ---- CHANGE ----
    if 'change_accuracy' in composite_metrics:
        print("CHANGE (Binary: Maintain vs Changed):")
        print(f"  Accuracy: {composite_metrics['change_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {composite_metrics['change_balanced_accuracy']:.4f}")
        print(f"  F1 (binary, changed): {composite_metrics.get('change_f1_binary', 0):.4f}")
        print(f"  Precision (changed): {composite_metrics.get('change_precision', 0):.4f}")
        print(f"  Recall (changed): {composite_metrics.get('change_recall', 0):.4f}")
        ch_auc = composite_metrics.get('change_auc', np.nan)
        if not np.isnan(ch_auc):
            print(f"  AUC: {ch_auc:.4f}")
        print(f"  Support: {composite_metrics['change_support']} (Changed: {composite_metrics['change_n_changed']}, Maintain: {composite_metrics['change_n_maintain']})")

        if 'change_confusion_matrix' in composite_metrics:
            cm_ch = np.array(composite_metrics['change_confusion_matrix'])
            print(f"  Confusion Matrix:")
            print(f"    Predicted ->")
            print(f"    Actual v   Maintain  Changed")
            print(f"    " + "-" * 30)
            print(f"    Maintain  {cm_ch[0,0]:>7}  {cm_ch[0,1]:>7}")
            print(f"    Changed   {cm_ch[1,0]:>7}  {cm_ch[1,1]:>7}")

        if 'change_accuracy_drug' in composite_metrics:
            print(f"\n  Drug-Conditioned (Drug Tokens Only):")
            print(f"    Accuracy: {composite_metrics['change_accuracy_drug']:.4f}")
            print(f"    Balanced Accuracy: {composite_metrics['change_balanced_accuracy_drug']:.4f}")
            print(f"    F1 (binary, changed): {composite_metrics.get('change_f1_binary_drug', 0):.4f}")
            print(f"    Precision (changed): {composite_metrics.get('change_precision_drug', 0):.4f}")
            print(f"    Recall (changed): {composite_metrics.get('change_recall_drug', 0):.4f}")
            ch_auc_drug = composite_metrics.get('change_auc_drug', np.nan)
            if not np.isnan(ch_auc_drug):
                print(f"    AUC: {ch_auc_drug:.4f}")
            print(f"    Support: {composite_metrics['change_support_drug']} (Changed: {composite_metrics['change_n_changed_drug']}, Maintain: {composite_metrics['change_n_maintain_drug']})")
            if 'change_confusion_matrix_drug' in composite_metrics:
                cm_ch_drug = np.array(composite_metrics['change_confusion_matrix_drug'])
                print(f"    Confusion Matrix (Drug Tokens Only):")
                print(f"      Predicted ->")
                print(f"      Actual v   Maintain  Changed")
                print(f"      " + "-" * 30)
                print(f"      Maintain  {cm_ch_drug[0,0]:>7}  {cm_ch_drug[0,1]:>7}")
                print(f"      Changed   {cm_ch_drug[1,0]:>7}  {cm_ch_drug[1,1]:>7}")

    # ---- SHIFT ----
    if 'shift_accuracy' in composite_metrics:
        print("SHIFT:")
        print(f"  Accuracy: {composite_metrics['shift_accuracy']:.4f}")
        if 'shift_balanced_accuracy' in composite_metrics:
            print(f"  Balanced Accuracy: {composite_metrics['shift_balanced_accuracy']:.4f}")
        if 'shift_f1_macro' in composite_metrics:
            print(f"  F1 (macro): {composite_metrics['shift_f1_macro']:.4f}")
        if 'shift_f1_micro' in composite_metrics:
            print(f"  F1 (micro): {composite_metrics['shift_f1_micro']:.4f}")
        if 'shift_f1_weighted' in composite_metrics:
            print(f"  F1 (weighted): {composite_metrics['shift_f1_weighted']:.4f}")
        if 'shift_precision_macro' in composite_metrics:
            print(f"  Precision (macro): {composite_metrics['shift_precision_macro']:.4f}")
        if 'shift_recall_macro' in composite_metrics:
            print(f"  Recall (macro): {composite_metrics['shift_recall_macro']:.4f}")
        if 'shift_support' in composite_metrics:
            print(f"  Support: {composite_metrics['shift_support']}")

        # SHIFT AUC
        s_auc_m = composite_metrics.get('shift_auc_macro', np.nan)
        s_auc_w = composite_metrics.get('shift_auc_weighted', np.nan)
        if not np.isnan(s_auc_m):
            print(f"  AUC (macro, OVR): {s_auc_m:.4f}")
        if not np.isnan(s_auc_w):
            print(f"  AUC (weighted, OVR): {s_auc_w:.4f}")
        if 'shift_per_class_auc' in composite_metrics:
            class_name_map = {1: 'Decrease', 2: 'Maintain', 3: 'Increase'}
            print(f"  Per-Class AUC (OVR):")
            for cls, auc_val in sorted(composite_metrics['shift_per_class_auc'].items()):
                name = class_name_map.get(int(cls), f'Class {cls}')
                if not np.isnan(auc_val):
                    print(f"    {name} (Class {cls}): {auc_val:.4f}")
                else:
                    print(f"    {name} (Class {cls}): N/A")

        # Confusion Matrix
        if 'shift_confusion_matrix' in composite_metrics and 'shift_confusion_matrix_classes' in composite_metrics:
            cm = np.array(composite_metrics['shift_confusion_matrix'])
            classes = composite_metrics['shift_confusion_matrix_classes']
            print(f"  Confusion Matrix:")
            print(f"    NOTE: Values shown are RAW classes (1=Dec, 2=Maint, 3=Inc)")
            print(f"    Predicted ->")
            header = "    Actual v   " + "  ".join([f"{int(c):>5}" for c in classes])
            print(header)
            print("    " + "-" * 30)
            for i, cls in enumerate(classes):
                row_str = f"    {int(cls):>5} " + "  ".join([f"{int(cm[i, j]):>5}" for j in range(len(classes))])
                print(row_str)
            if 'shift_per_class_metrics' in composite_metrics:
                print(f"  Per-Class Metrics:")
                for cls, metrics in sorted(composite_metrics['shift_per_class_metrics'].items()):
                    print(f"    Class {cls}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                          f"F1={metrics['f1']:.4f}, Support={metrics['support']}")

        # Drug-conditioned SHIFT
        if 'shift_accuracy_drug_cond' in composite_metrics:
            print(f"\n  Drug-Conditioned (Drug Tokens Only, 1279-1289):")
            print(f"    Accuracy: {composite_metrics['shift_accuracy_drug_cond']:.4f}")
            if 'shift_balanced_accuracy_drug_cond' in composite_metrics:
                print(f"    Balanced Accuracy: {composite_metrics['shift_balanced_accuracy_drug_cond']:.4f}")
            for k_name in ['f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro', 'recall_macro']:
                k_full = f'shift_{k_name}_drug_cond'
                if k_full in composite_metrics:
                    print(f"    {k_name.replace('_', ' ').title()}: {composite_metrics[k_full]:.4f}")
            if 'shift_support_drug_cond' in composite_metrics:
                print(f"    Support: {composite_metrics['shift_support_drug_cond']}")

            # Drug-conditioned SHIFT AUC
            s_auc_m_d = composite_metrics.get('shift_auc_macro_drug_cond', np.nan)
            s_auc_w_d = composite_metrics.get('shift_auc_weighted_drug_cond', np.nan)
            if not np.isnan(s_auc_m_d):
                print(f"    AUC (macro, OVR): {s_auc_m_d:.4f}")
            if not np.isnan(s_auc_w_d):
                print(f"    AUC (weighted, OVR): {s_auc_w_d:.4f}")
            if 'shift_per_class_auc_drug_cond' in composite_metrics:
                class_name_map = {1: 'Decrease', 2: 'Maintain', 3: 'Increase'}
                print(f"    Per-Class AUC (OVR):")
                for cls, auc_val in sorted(composite_metrics['shift_per_class_auc_drug_cond'].items()):
                    name = class_name_map.get(int(cls), f'Class {cls}')
                    if not np.isnan(auc_val):
                        print(f"      {name} (Class {cls}): {auc_val:.4f}")
                    else:
                        print(f"      {name} (Class {cls}): N/A")

            if 'shift_confusion_matrix_drug_cond' in composite_metrics and 'shift_confusion_matrix_drug_cond_classes' in composite_metrics:
                cm_drug = np.array(composite_metrics['shift_confusion_matrix_drug_cond'])
                classes_drug = composite_metrics['shift_confusion_matrix_drug_cond_classes']
                print(f"    Confusion Matrix (Drug-Conditioned, Drug Tokens Only):")
                print(f"      NOTE: Values shown are RAW classes (1=Dec, 2=Maint, 3=Inc)")
                print(f"      Predicted ->")
                header = "      Actual v   " + "  ".join([f"{int(c):>5}" for c in classes_drug])
                print(header)
                print("      " + "-" * 30)
                for i, cls in enumerate(classes_drug):
                    row_str = f"      {int(cls):>5} " + "  ".join([f"{int(cm_drug[i, j]):>5}" for j in range(len(classes_drug))])
                    print(row_str)
                if 'shift_per_class_metrics_drug_cond' in composite_metrics:
                    print(f"    Per-Class Metrics (Drug-Conditioned):")
                    for cls, metrics in sorted(composite_metrics['shift_per_class_metrics_drug_cond'].items()):
                        print(f"      Class {cls}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                              f"F1={metrics['f1']:.4f}, Support={metrics['support']}")

    # ---- TOTAL ----
    field = 'total'
    if f'{field}_mae' in composite_metrics:
        print(f"{field.upper()}:")
        print(f"  MAE: {composite_metrics[f'{field}_mae']:.4f}")
        if f'{field}_rmse' in composite_metrics:
            print(f"  RMSE: {composite_metrics[f'{field}_rmse']:.4f}")
        if f'{field}_median_ae' in composite_metrics:
            print(f"  Median AE: {composite_metrics[f'{field}_median_ae']:.4f}")
        if f'{field}_r2' in composite_metrics and not np.isnan(composite_metrics[f'{field}_r2']):
            print(f"  R2: {composite_metrics[f'{field}_r2']:.4f}")
        if f'{field}_mae_pos' in composite_metrics:
            print(f"  MAE (target>0): {composite_metrics[f'{field}_mae_pos']:.4f} (n={composite_metrics.get(f'{field}_support_pos', 'NA')})")
        if 'total_mae_drug_cond' in composite_metrics:
            print(f"\n  Drug-Conditioned (Drug Tokens Only, 1279-1289):")
            print(f"    MAE: {composite_metrics['total_mae_drug_cond']:.4f}")
            if 'total_rmse_drug_cond' in composite_metrics:
                print(f"    RMSE: {composite_metrics['total_rmse_drug_cond']:.4f}")
            if 'total_median_ae_drug_cond' in composite_metrics:
                print(f"    Median AE: {composite_metrics['total_median_ae_drug_cond']:.4f}")
            if 'total_r2_drug_cond' in composite_metrics and not np.isnan(composite_metrics['total_r2_drug_cond']):
                print(f"    R2: {composite_metrics['total_r2_drug_cond']:.4f}")
            if 'total_mean_target_drug_cond' in composite_metrics:
                print(f"    Mean Target: {composite_metrics['total_mean_target_drug_cond']:.4f}")
            if 'total_mean_pred_drug_cond' in composite_metrics:
                print(f"    Mean Prediction: {composite_metrics['total_mean_pred_drug_cond']:.4f}")
            if 'total_support_drug_cond' in composite_metrics:
                print(f"    Support: {composite_metrics['total_support_drug_cond']}")
    print("=" * 60)


def evaluate_auc_pipeline(
    model,
    d100k,
    output_path,
    labels_df,
    model_type='modern',
    evaluate_composite=True,
    diseases_of_interest=None,
    filter_min_total=100,
    disease_chunk_size=200,
    age_groups=np.arange(40, 80, 5),
    offset=0.1,
    batch_size=64,
    device="cpu",
    seed=1337,
    n_bootstrap=1,
    meta_info={},
    train_valid_tokens=None,
):
    assert n_bootstrap > 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    config_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else model.config.data_vocab_size if hasattr(model.config, 'data_vocab_size') else 1290
    vocab_size = config_vocab_size

    if 'index' in labels_df.columns:
        max_label_index = labels_df['index'].max()
        if max_label_index >= vocab_size - 1:
            print(f"Warning: labels contain index {max_label_index}, but model vocab_size is {vocab_size}.")

    if diseases_of_interest is None:
        diseases_of_interest = get_common_diseases(labels_df, filter_min_total)
    diseases_of_interest = [d for d in diseases_of_interest if 0 <= d < vocab_size]

    if model_type == 'composite':
        target_data_np = d100k[4].cpu().detach().numpy()
    else:
        target_data_np = d100k[2].cpu().detach().numpy()

    actual_tokens_in_data = set(np.unique(target_data_np).tolist())
    actual_tokens_in_data = {t for t in actual_tokens_in_data if t >= 0}

    diseases_before_filter = len(diseases_of_interest)
    diseases_of_interest = [d for d in diseases_of_interest if d in actual_tokens_in_data]
    diseases_filtered_eval = diseases_before_filter - len(diseases_of_interest)
    if diseases_filtered_eval > 0:
        print(f"Filtered out {diseases_filtered_eval} diseases not present in evaluation data")

    if train_valid_tokens is not None:
        diseases_before_train_filter = len(diseases_of_interest)
        diseases_of_interest = [d for d in diseases_of_interest if d in train_valid_tokens]
        diseases_filtered_train = diseases_before_train_filter - len(diseases_of_interest)
        if diseases_filtered_train > 0:
            print(f"Filtered out {diseases_filtered_train} diseases not present in train data")

    if len(diseases_of_interest) == 0:
        raise ValueError(f"No valid diseases found.")

    print(f"Evaluating {len(diseases_of_interest)} diseases (vocab_size={vocab_size}, actual unique tokens in eval data={len(actual_tokens_in_data)})")

    num_chunks = (len(diseases_of_interest) + disease_chunk_size - 1) // disease_chunk_size
    diseases_chunks = np.array_split(diseases_of_interest, num_chunks)

    if model_type == 'composite':
        data_tokens = d100k[0].cpu().detach().numpy()
        ages = d100k[3].cpu().detach().numpy()
        target_data = d100k[4].cpu().detach().numpy()
        target_ages = d100k[7].cpu().detach().numpy()
        d = [data_tokens, ages, target_data, target_ages]
    else:
        tokens = d100k[0].cpu().detach().numpy()
        ages = d100k[1].cpu().detach().numpy()
        targets = d100k[2].cpu().detach().numpy()
        target_ages = d100k[3].cpu().detach().numpy()
        d = [tokens, ages, targets, target_ages]

    pred_idx_precompute = (d[1][:, :, np.newaxis] <= d[3][:, np.newaxis, :] - offset).sum(1) - 1

    all_aucs = []
    tqdm_options = {"desc": "Processing disease chunks", "total": len(diseases_chunks)}
    for disease_chunk_idx, diseases_chunk in tqdm(enumerate(diseases_chunks), **tqdm_options):
        diseases_chunk = np.array(diseases_chunk)
        valid_mask = (diseases_chunk >= 0) & (diseases_chunk < vocab_size)
        diseases_chunk = diseases_chunk[valid_mask].tolist()
        if len(diseases_chunk) == 0:
            continue

        p100k = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            if model_type == 'composite':
                x_data, x_shift, x_total, x_ages = d100k[0], d100k[1], d100k[2], d100k[3]
                num_batches = (x_data.shape[0] + batch_size - 1) // batch_size
                for batch_idx in tqdm(range(num_batches), desc=f"Model inference, chunk {disease_chunk_idx}"):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, x_data.shape[0])
                    batch_x_data = x_data[start_idx:end_idx].to(device)
                    batch_x_shift = x_shift[start_idx:end_idx].to(device)
                    batch_x_total = x_total[start_idx:end_idx].to(device)
                    batch_x_ages = x_ages[start_idx:end_idx].to(device)
                    outputs = model(batch_x_data, batch_x_shift, batch_x_total, batch_x_ages)[0]
                    data_logits = outputs['data'].cpu().detach().numpy()
                    p100k.append(data_logits[:, :, diseases_chunk].astype("float16"))
            else:
                x, a = d100k[0], d100k[1]
                num_batches = (x.shape[0] + batch_size - 1) // batch_size
                for batch_idx in tqdm(range(num_batches), desc=f"Model inference, chunk {disease_chunk_idx}"):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, x.shape[0])
                    batch_x = x[start_idx:end_idx].to(device)
                    batch_a = a[start_idx:end_idx].to(device)
                    outputs = model(batch_x, batch_a)[0].cpu().detach().numpy()
                    p100k.append(outputs[:, :, diseases_chunk].astype("float16"))

        if len(p100k) == 0:
            continue
        p100k = np.vstack(p100k)

        for j, k in tqdm(list(enumerate(diseases_chunk)), desc=f"Processing diseases in chunk {disease_chunk_idx}"):
            out = get_calibration_auc(j, k, d, p100k, diseases_chunk, age_groups=age_groups, offset=offset,
                                      precomputed_idx=pred_idx_precompute, n_bootstrap=n_bootstrap, use_delong=True)
            if out is None:
                continue
            for out_item in out:
                all_aucs.append(out_item)

    df_auc_unpooled = pd.DataFrame(all_aucs)
    for key, value in meta_info.items():
        df_auc_unpooled[key] = value

    if 'index' in labels_df.columns:
        labels_df_subset = labels_df[['index']].copy()
        if 'name' in labels_df.columns:
            labels_df_subset['name'] = labels_df['name']
        labels_df_subset['shifted_token'] = labels_df_subset['index'] + 1
        df_auc_unpooled_merged = df_auc_unpooled.merge(labels_df_subset, left_on="token", right_on="shifted_token", how="inner")
    else:
        df_auc_unpooled_merged = df_auc_unpooled.copy()

    def aggregate_age_brackets_delong(group):
        n = len(group)
        valid_aucs = group['auc_delong'].dropna()
        if len(valid_aucs) == 0:
            mean = np.nan; var = np.nan
            status = group['status'].iloc[0] if 'status' in group.columns else 'unknown'
        else:
            mean = valid_aucs.mean()
            valid_vars = group.loc[valid_aucs.index, 'auc_variance_delong']
            var = valid_vars.sum() / (len(valid_vars)**2) if len(valid_vars) > 0 else np.nan
            status = 'ok'
        if isinstance(var, np.ndarray):
            var = var.item() if var.size == 1 else float(var[0, 0]) if var.ndim > 0 else float(var)
        elif not np.isnan(var):
            var = float(var)
        return pd.Series({'auc': mean, 'auc_variance_delong': var, 'n_samples': n,
                          'n_diseased': group['n_diseased'].sum(), 'n_healthy': group['n_healthy'].sum(), 'status': status})

    print('Using DeLong method to calculate AUC confidence intervals..')
    df_auc = df_auc_unpooled.groupby(["token"]).apply(aggregate_age_brackets_delong, include_groups=False).reset_index()

    if 'index' in labels_df.columns:
        labels_df_for_merge = labels_df.copy()
        labels_df_for_merge['shifted_token'] = labels_df_for_merge['index'] + 1
        df_auc_merged = df_auc.merge(labels_df_for_merge, left_on="token", right_on="shifted_token", how="inner")
    else:
        df_auc_merged = df_auc.copy()

    # Evaluate composite fields
    composite_metrics = None
    if model_type == 'composite' and evaluate_composite:
        print("\nEvaluating composite fields (SHIFT, CHANGE, TOTAL)...")
        composite_metrics = evaluate_composite_fields(model, d100k, batch_size=batch_size, device=device)
        _print_composite_metrics(composite_metrics)

        if output_path is not None:
            import json
            with open(f"{output_path}/composite_metrics.json", 'w') as f:
                json_metrics = {}
                for k, v in composite_metrics.items():
                    if isinstance(v, dict):
                        json_metrics[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating, int, float)) else v2 for k2, v2 in v.items()}
                    elif isinstance(v, (np.integer, np.floating, np.ndarray)):
                        json_metrics[k] = v.tolist() if isinstance(v, np.ndarray) else float(v)
                    elif isinstance(v, (int, float, bool, str)):
                        json_metrics[k] = v
                    else:
                        try: json_metrics[k] = float(v)
                        except: json_metrics[k] = str(v)
                json.dump(json_metrics, f, indent=2)
            print(f"Composite metrics saved to {output_path}/composite_metrics.json")

    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)
        df_auc_merged.to_parquet(f"{output_path}/df_both.parquet", index=False)
        df_auc_unpooled_merged.to_parquet(f"{output_path}/df_auc_unpooled.parquet", index=False)

    return df_auc_unpooled_merged, df_auc_merged, composite_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate AUC")
    parser.add_argument("--input_path", type=str, default="../data", help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default="results", help="Path to the output")
    parser.add_argument("--model_ckpt_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--model_type", type=str, default='modern', choices=['modern', 'composite'])
    parser.add_argument("--no_event_token_rate", type=int, default=5)
    parser.add_argument("--health_token_replacement_prob", default=0.0, type=float)
    parser.add_argument("--dataset_subset_size", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=1)
    parser.add_argument("--filter_min_total", type=int, default=0)
    parser.add_argument("--disease_chunk_size", type=int, default=200)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=80)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--data_files", type=str, default=None)
    parser.add_argument("--train_data_file", type=str, default="kr_train.bin")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_type = args.model_type
    no_event_token_rate = args.no_event_token_rate
    dataset_subset_size = args.dataset_subset_size

    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    print(device)
    seed = 1337

    ckpt_path = args.model_ckpt_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]

    use_moe = model_args.get('use_moe', False)
    num_experts = model_args.get('num_experts', 0)
    experts_per_token = model_args.get('experts_per_token', 0)

    if model_type == 'composite':
        conf = CompositeDelphiConfig(**model_args)
        model = CompositeDelphi(conf)
    else:
        conf = ModernDelphiConfig(**model_args)
        model = ModernDelphi(conf)

    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    print(f"\n{'='*60}")
    print(f"Model Architecture Info:")
    print(f"  Model type: {model_type}")
    print(f"  Use MoE: {use_moe}")
    if use_moe:
        print(f"  Number of experts: {num_experts}")
        print(f"  Experts per token: {experts_per_token}")
    print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"{'='*60}\n")

    if args.labels_path:
        labels_df = pd.read_csv(args.labels_path, header=None, usecols=[0], names=['name'])
        labels_df['index'] = range(len(labels_df))
    else:
        labels_path = f"{input_path}/labels.csv"
        if Path(labels_path).exists():
            labels_df = pd.read_csv(labels_path, header=None, usecols=[0], names=['name'])
            labels_df['index'] = range(len(labels_df))
        else:
            print(f"Warning: labels file not found at {labels_path}. Creating minimal labels DataFrame.")
            labels_df = pd.DataFrame({'index': range(2000), 'name': [f'token_{i}' for i in range(2000)]})

    if args.data_files:
        data_files_list = []
        for f in args.data_files.split(','):
            f = f.strip()
            if f:
                f_lower = f.lower()
                if 'ukb' in f_lower and 'extval' in f_lower: prefix = 'extval_ukb'
                elif 'jmdc' in f_lower and 'extval' in f_lower: prefix = 'extval_jmdc'
                elif 'extval' in f_lower: prefix = 'extval'
                elif 'val' in f_lower: prefix = 'val'
                elif 'test' in f_lower: prefix = 'test'
                else: prefix = Path(f).stem
                data_files_list.append((f, prefix))
    else:
        data_files_list = [
            ("kr_val.bin", "val"),
            ("kr_test.bin", "test"),
            ("JMDC_extval.bin", "extval_jmdc"),
            ("UKB_extval.bin", "extval_ukb"),
        ]

    base_meta_info = {'model_type': model_type, 'use_moe': use_moe}
    if use_moe:
        base_meta_info['num_experts'] = num_experts
        base_meta_info['experts_per_token'] = experts_per_token
    if 'iter_num' in checkpoint:
        base_meta_info['checkpoint_iter'] = checkpoint['iter_num']
    if 'best_val_loss' in checkpoint:
        base_meta_info['checkpoint_val_loss'] = checkpoint['best_val_loss']

    composite_dtype = np.dtype([
        ('ID', np.uint32), ('AGE', np.uint32), ('DATA', np.uint32),
        ('SHIFT', np.uint32), ('TOTAL', np.uint32)
    ])

    # Load train data for valid token filtering
    train_data_path = f"{input_path}/{args.train_data_file}"
    train_valid_tokens = None
    train_prefix = args.train_data_file.split('_')[0] if '_' in args.train_data_file else None

    if Path(train_data_path).exists():
        print(f"\nLoading train data to filter valid tokens: {train_data_path}")
        if model_type == 'composite':
            train_data_raw = np.fromfile(train_data_path, dtype=composite_dtype)
            train_raw_tokens = np.unique(train_data_raw['DATA'])
            train_valid_tokens = set((train_raw_tokens + 1).tolist())
        else:
            train_data_raw = np.fromfile(train_data_path, dtype=np.uint32).reshape(-1, 3)
            train_raw_tokens = np.unique(train_data_raw[:, 2])
            train_valid_tokens = set((train_raw_tokens + 1).tolist())
        print(f"  Train data contains {len(train_valid_tokens)} unique tokens (after +1 shift)")

        drug_token_names = {
            1279: 'Metformin', 1280: 'Sulfonylurea', 1281: 'DPP-4', 1282: 'Insulin',
            1283: 'Meglitinide', 1284: 'Thiazolidinedione', 1285: 'Alpha-glucosidase',
            1286: 'GLP-1', 1287: 'SGLT-2', 1288: 'Other', 1289: 'Death'
        }
        print("  Drug tokens in train data:")
        for token, name in drug_token_names.items():
            status = "V" if token in train_valid_tokens else "X"
            print(f"    {status} {name} ({token})")
    else:
        print(f"\n[WARNING] Train data not found at {train_data_path}.")
        train_prefix = None

    all_results = {}
    for data_filename, prefix in data_files_list:
        data_filepath = f"{input_path}/{data_filename}"
        if not Path(data_filepath).exists():
            print(f"\n[WARNING] Skipping {data_filename}: file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {data_filename} (prefix: {prefix})")
        print(f"{'='*60}\n")

        if model_type == 'composite':
            data = np.fromfile(data_filepath, dtype=composite_dtype)
            data_p2i = get_p2i_composite(data)
        else:
            data = np.fromfile(data_filepath, dtype=np.uint32).reshape(-1, 3).astype(np.int64)
            data_p2i = get_p2i(data)

        current_subset_size = dataset_subset_size
        if current_subset_size == -1:
            current_subset_size = len(data_p2i)
        else:
            current_subset_size = min(current_subset_size, len(data_p2i))

        print(f"Using {current_subset_size} patients for evaluation (out of {len(data_p2i)} total)")

        np.random.seed(seed)
        patient_indices = np.random.choice(len(data_p2i), size=current_subset_size, replace=False)
        patient_indices = sorted(patient_indices)

        if model_type == 'composite':
            d100k = get_batch_composite(
                patient_indices, data, data_p2i, select="left", block_size=args.block_size,
                device=device, padding="random", no_event_token_rate=no_event_token_rate,
                apply_token_shift=False,
            )
        else:
            d100k = get_batch(
                patient_indices, data, data_p2i, select="left", block_size=args.block_size,
                device=device, padding="random", no_event_token_rate=no_event_token_rate,
            )

        meta_info = base_meta_info.copy()
        meta_info['data_source'] = data_filename
        meta_info['data_prefix'] = prefix

        result = evaluate_auc_pipeline(
            model, d100k, output_path=None, labels_df=labels_df, model_type=model_type,
            evaluate_composite=(prefix != 'extval_ukb'),
            diseases_of_interest=None, filter_min_total=args.filter_min_total,
            disease_chunk_size=args.disease_chunk_size, batch_size=args.eval_batch_size,
            device=device, seed=seed, n_bootstrap=args.n_bootstrap, meta_info=meta_info,
            train_valid_tokens=None,
        )

        if model_type == 'composite':
            df_auc_unpooled, df_auc_merged, composite_metrics = result
            if composite_metrics is None:
                composite_metrics = {}
            if df_auc_merged is not None and not df_auc_merged.empty and 'auc' in df_auc_merged.columns:
                auc_values = df_auc_merged['auc'].dropna()
                if not auc_values.empty:
                    composite_metrics['auc_mean'] = float(auc_values.mean())
                    composite_metrics['auc_median'] = float(auc_values.median())
                    composite_metrics['auc_min'] = float(auc_values.min())
                    composite_metrics['auc_max'] = float(auc_values.max())
                    composite_metrics['auc_std'] = float(auc_values.std())
                    composite_metrics['n_diseases_auc'] = int(len(auc_values))
                    print(f"\n[AUC Statistics] (Next Disease Prediction)")
                    print(f"  Mean:   {composite_metrics['auc_mean']:.4f}")
                    print(f"  Median: {composite_metrics['auc_median']:.4f}")
                    print(f"  Min/Max: {composite_metrics['auc_min']:.4f} / {composite_metrics['auc_max']:.4f}")
        else:
            df_auc_unpooled, df_auc_merged = result[:2]
            composite_metrics = None

        if output_path is not None:
            df_auc_merged.to_parquet(f"{output_path}/{prefix}_df_both.parquet", index=False)
            df_auc_unpooled.to_parquet(f"{output_path}/{prefix}_df_auc_unpooled.parquet", index=False)
            df_auc_merged.to_csv(f"{output_path}/{prefix}_df_both.csv", index=False)
            df_auc_unpooled.to_csv(f"{output_path}/{prefix}_df_auc_unpooled.csv", index=False)
            if composite_metrics:
                import json
                with open(f"{output_path}/{prefix}_composite_metrics.json", 'w') as f:
                    json_metrics = {}
                    for k, v in composite_metrics.items():
                        if isinstance(v, dict):
                            json_metrics[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating, int, float)) else v2 for k2, v2 in v.items()}
                        elif isinstance(v, (np.integer, np.floating, np.ndarray)):
                            json_metrics[k] = v.tolist() if isinstance(v, np.ndarray) else float(v)
                        elif isinstance(v, (int, float, bool, str)):
                            json_metrics[k] = v
                        else:
                            try: json_metrics[k] = float(v)
                            except: json_metrics[k] = str(v)
                    json.dump(json_metrics, f, indent=2)

        all_results[prefix] = {
            'df_auc_unpooled': df_auc_unpooled, 'df_auc_merged': df_auc_merged,
            'composite_metrics': composite_metrics, 'data_filename': data_filename,
        }
        print(f"\n[{prefix.upper()}] Evaluation completed!")
        print(f"  Total diseases evaluated: {len(df_auc_merged)}")
        if composite_metrics:
            print(f"  Composite field metrics saved to {output_path}/{prefix}_composite_metrics.json")

    print(f"\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    for prefix, result_data in all_results.items():
        print(f"\n[{prefix.upper()}] {result_data['data_filename']}:")
        print(f"  - {prefix}_df_both.parquet / .csv")
        print(f"  - {prefix}_df_auc_unpooled.parquet / .csv")
        if result_data['composite_metrics']:
            print(f"  - {prefix}_composite_metrics.json")
        print(f"  - Diseases evaluated: {len(result_data['df_auc_merged'])}")


if __name__ == "__main__":
    main()
