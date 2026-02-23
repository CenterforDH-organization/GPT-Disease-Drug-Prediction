"""
performance.py - Shift / Total prediction performance analysis
===============================================================
Contains CompositeModelAnalyzer: collects predictions from the model
and produces separate figures for shift classification and total regression.

Moved from composite_model_analysis.py into figutils/ so notebooks can
import via ``from figutils.performance import CompositeModelAnalyzer``.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score,
)
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class CompositeModelAnalyzer:
    """
    Analyzer for Composite Delphi model predictions.
    Handles both shift (classification) and total (regression) tasks.
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        self.predictions = {'shift': [], 'total': [], 'disease': []}
        self.targets     = {'shift': [], 'total': [], 'disease': []}
        self.attention_weights = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_predictions(self, dataloader, max_batches=None):
        """Run the model on *dataloader* and store predictions + targets."""
        print("Collecting predictions...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                if len(batch) != 8:
                    print(f"Warning: unexpected batch length {len(batch)}, skipping")
                    continue

                x_data, x_shift, x_total, x_ages = [b.to(self.device) for b in batch[:4]]
                y_data, y_shift, y_total, y_ages = [b.to(self.device) for b in batch[4:]]

                logits, loss, att = self.model(
                    x_data, x_shift, x_total, x_ages,
                    targets_data=y_data,
                    targets_shift=y_shift,
                    targets_total=y_total,
                    targets_age=y_ages,
                )

                # Use drug-conditioned heads when available
                shift_logits = logits.get('shift_drug_cond', logits['shift'])
                total_pred   = logits.get('total_drug_cond', logits['total'])

                self.predictions['disease'].append(logits['data'].cpu())
                self.predictions['shift'].append(shift_logits.cpu())
                self.predictions['total'].append(total_pred.cpu())

                self.targets['disease'].append(y_data.cpu())
                self.targets['shift'].append(y_shift.cpu())
                self.targets['total'].append(y_total.cpu())

                if att is not None:
                    self.attention_weights.append(att.cpu())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  {batch_idx + 1} batches processed")

        self._concatenate()
        print("Prediction collection complete!")

    # ------------------------------------------------------------------
    def _concatenate(self):
        """Pad variable-length batches and concatenate."""
        for key in ('disease', 'shift', 'total'):
            preds = self.predictions[key]
            tgts  = self.targets[key]
            if not preds:
                continue

            max_len = max(p.shape[1] for p in preds)

            def _pad_list(tensors, max_len):
                out = []
                for t in tensors:
                    if t.shape[1] < max_len:
                        pad_shape = list(t.shape)
                        pad_shape[1] = max_len - t.shape[1]
                        t = torch.cat([t, torch.zeros(*pad_shape, dtype=t.dtype)], dim=1)
                    out.append(t)
                return torch.cat(out, dim=0)

            self.predictions[key] = _pad_list(preds, max_len)
            self.targets[key]     = _pad_list(tgts,  max_len)

    # ==================================================================
    # Shift classification figure (standalone)
    # ==================================================================
    def visualize_shift_performance(self, save_path=None):
        """
        6-panel figure for drug-dose shift classification.

        Only evaluates drug events (shift label ∈ {2, 3, 4}).
        If model predicts 0(Pad) or 1(Non-Drug) for a drug event,
        it counts as a misclassification (not clipped).

        Returns
        -------
        report : dict   (sklearn classification_report + 'accuracy')
        """
        shift_logits = self.predictions['shift']
        shift_target = self.targets['shift']
        if shift_logits is None or len(shift_logits) == 0:
            print("No shift predictions available")
            return None

        shift_pred = torch.argmax(shift_logits, dim=-1).numpy().flatten()
        shift_true = shift_target.numpy().flatten()

        # ── Filter: keep only drug events (true label ∈ {2,3,4}) ──
        # After +1 token shift: 0=pad, 1=non-drug, 2=decrease, 3=maintain, 4=increase
        mask = np.isin(shift_true, [2, 3, 4])
        shift_pred = shift_pred[mask]
        shift_true = shift_true[mask]

        if len(shift_true) == 0:
            print("No drug-related shift predictions after filtering")
            return None

        # Diagnostic: show what model actually predicts (including invalid classes)
        pred_unique, pred_counts = np.unique(shift_pred, return_counts=True)
        print(f"[DEBUG] Raw prediction distribution:")
        label_map = {0: 'Pad', 1: 'Non-Drug', 2: 'Decrease', 3: 'Maintain', 4: 'Increase'}
        for v, c in zip(pred_unique, pred_counts):
            print(f"  {label_map.get(v, f'?{v}'):>10s} (idx={v}): {c:>8,} ({c/len(shift_pred)*100:.1f}%)")

        # ── Fixed labels: Decrease / Maintain / Increase ──
        class_labels = [2, 3, 4]
        class_names  = ['Decrease', 'Maintain', 'Increase']
        class_colors = ['salmon', 'lightgreen', 'orange']

        print(f"Shift analysis — {len(shift_true):,} drug events")
        print(f"  Classes: {dict(zip(class_names, [np.sum(shift_true==c) for c in class_labels]))}")

        # ── figure ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug Dose Shift Prediction Analysis (Drug Events Only)',
                     fontsize=16, fontweight='bold')

        # 1. Confusion matrix (counts)
        cm = confusion_matrix(shift_true, shift_pred, labels=class_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True'); axes[0, 0].set_xlabel('Predicted')

        # 2. Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title('Normalized Confusion Matrix')
        axes[0, 1].set_ylabel('True'); axes[0, 1].set_xlabel('Predicted')

        # 3. Class distribution
        dist_counts = [np.sum(shift_true == c) for c in class_labels]
        axes[0, 2].bar(class_names, dist_counts, color='skyblue')
        axes[0, 2].set_title('True Class Distribution'); axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 4. Per-class accuracy
        per_class_acc = cm_norm.diagonal()
        axes[1, 0].bar(class_names, per_class_acc, color=class_colors)
        axes[1, 0].set_title('Per-Class Accuracy'); axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1); axes[1, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(per_class_acc):
            axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

        # 5. Precision / Recall / F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            shift_true, shift_pred, average=None, labels=class_labels)
        x = np.arange(len(class_names)); w = 0.25
        axes[1, 1].bar(x - w, prec, w, label='Precision', color='steelblue')
        axes[1, 1].bar(x,     rec,  w, label='Recall',    color='darkorange')
        axes[1, 1].bar(x + w, f1,   w, label='F1-Score',  color='green')
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(class_names)
        axes[1, 1].set_title('Precision / Recall / F1'); axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1); axes[1, 1].tick_params(axis='x', rotation=45)

        # 6. Text report
        report = classification_report(shift_true, shift_pred,
                                       target_names=class_names,
                                       labels=class_labels, output_dict=True)
        overall_acc = accuracy_score(shift_true, shift_pred)
        report['accuracy'] = overall_acc

        txt  = f"Overall Accuracy: {overall_acc:.2%}\n\n"
        txt += f"Macro Avg:\n"
        txt += f"  Precision: {report['macro avg']['precision']:.2%}\n"
        txt += f"  Recall:    {report['macro avg']['recall']:.2%}\n"
        txt += f"  F1-Score:  {report['macro avg']['f1-score']:.2%}\n\n"
        txt += f"Weighted Avg:\n"
        txt += f"  Precision: {report['weighted avg']['precision']:.2%}\n"
        txt += f"  Recall:    {report['weighted avg']['recall']:.2%}\n"
        txt += f"  F1-Score:  {report['weighted avg']['f1-score']:.2%}"
        axes[1, 2].text(0.1, 0.5, txt, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 2].axis('off'); axes[1, 2].set_title('Classification Report')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        return report

    # ==================================================================
    # Total regression figure (standalone)
    # ==================================================================
    def visualize_total_performance(self, save_path=None):
        """
        6-panel figure for total-dosage regression.

        Returns
        -------
        dict with keys: n_samples, mse, rmse, mae, r2, mape
        """
        total_pred_all = self.predictions['total']
        total_true_all = self.targets['total']
        if total_pred_all is None or len(total_pred_all) == 0:
            print("No total predictions available")
            return None

        total_pred = total_pred_all.numpy().flatten()
        total_true = total_true_all.numpy().flatten()
        shift_true = self.targets['shift'].numpy().flatten()

        # Keep only drug events with valid positive totals
        mask = (shift_true != 0) & (shift_true != 1) & (total_true > 0) & (total_pred > 0)
        total_pred = total_pred[mask]
        total_true = total_true[mask]

        if len(total_true) == 0:
            print("No drug-related total predictions after filtering")
            return None

        print(f"Total analysis — {len(total_true):,} drug events")

        mse  = mean_squared_error(total_true, total_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(total_true, total_pred)
        r2   = r2_score(total_true, total_pred)
        mape = np.mean(np.abs((total_true - total_pred) / total_true)) * 100
        residuals = total_pred - total_true

        # ── figure ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug Total Dosage Prediction Analysis (Drug Events Only)',
                     fontsize=16, fontweight='bold')

        # 1. Scatter
        axes[0, 0].scatter(total_true, total_pred, alpha=0.3, s=20)
        axes[0, 0].plot([total_true.min(), total_true.max()],
                        [total_true.min(), total_true.max()],
                        'r--', lw=2, label='Perfect')
        axes[0, 0].set_xlabel('True'); axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predicted vs True'); axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals
        axes[0, 1].scatter(total_true, residuals, alpha=0.3, s=20)
        axes[0, 1].axhline(0, color='r', ls='--', lw=2)
        axes[0, 1].set_xlabel('True'); axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title('Residual Plot'); axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution comparison
        axes[0, 2].hist(total_true, bins=50, alpha=0.5, label='True', color='blue')
        axes[0, 2].hist(total_pred, bins=50, alpha=0.5, label='Predicted', color='orange')
        axes[0, 2].set_xlabel('Dosage'); axes[0, 2].set_ylabel('Freq')
        axes[0, 2].set_title('Distribution Comparison'); axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Error distribution
        axes[1, 0].hist(residuals, bins=50, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(0, color='r', ls='--', lw=2)
        axes[1, 0].set_xlabel('Error'); axes[1, 0].set_ylabel('Freq')
        axes[1, 0].set_title('Error Distribution'); axes[1, 0].grid(True, alpha=0.3)

        # 5. Absolute error vs true
        axes[1, 1].scatter(total_true, np.abs(residuals), alpha=0.3, s=20)
        axes[1, 1].set_xlabel('True'); axes[1, 1].set_ylabel('|Error|')
        axes[1, 1].set_title('Abs Error vs True'); axes[1, 1].grid(True, alpha=0.3)

        # 6. Metrics text
        txt  = "Regression Metrics:\n\n"
        txt += f"MSE:  {mse:.4f}\n"
        txt += f"RMSE: {rmse:.4f}\n"
        txt += f"MAE:  {mae:.4f}\n"
        txt += f"R²:   {r2:.4f}\n"
        txt += f"MAPE: {mape:.2f}%\n\n"
        txt += f"Samples: {len(total_true):,}\n"
        txt += f"Mean True: {total_true.mean():.2f}  Pred: {total_pred.mean():.2f}\n"
        txt += f"Std  True: {total_true.std():.2f}  Pred: {total_pred.std():.2f}"
        axes[1, 2].text(0.1, 0.5, txt, fontsize=12, family='monospace',
                        verticalalignment='center')
        axes[1, 2].axis('off'); axes[1, 2].set_title('Regression Metrics')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()

        return dict(n_samples=len(total_true), mse=mse, rmse=rmse,
                    mae=mae, r2=r2, mape=mape)

    # ==================================================================
    # Attention maps (convenience)
    # ==================================================================
    def visualize_attention_maps(self, sample_indices=(0, 1, 2),
                                 save_path=None):
        """Quick attention heatmap for a few samples."""
        if not self.attention_weights:
            print("No attention weights available")
            return

        n = min(len(sample_indices), len(self.attention_weights))
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        fig.suptitle('Attention Map Visualization', fontsize=16, fontweight='bold')

        for idx, si in enumerate(sample_indices[:n]):
            attn = self.attention_weights[si]
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().numpy()
            if len(attn.shape) == 4:
                attn = attn[0].mean(axis=0)
            elif len(attn.shape) == 3:
                attn = attn.mean(axis=0)
            im = axes[idx].imshow(attn, cmap='viridis', aspect='auto')
            axes[idx].set_title(f'Sample {si}')
            axes[idx].set_xlabel('Key'); axes[idx].set_ylabel('Query')
            plt.colorbar(im, ax=axes[idx])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()