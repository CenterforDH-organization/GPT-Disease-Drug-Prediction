"""
shap_viz.py - SHAP interaction heatmaps & analysis
=====================================================
All draw_* functions return a Figure and display inline.
Optionally pass save_path to persist to disk.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm

# =============================================================================
# Data Loading & Processing
# =============================================================================

def load_shap_data(shap_pickle_path, labels_csv_path):
    """
    Load SHAP data and create token mappings
    
    Parameters
    ----------
    shap_pickle_path : str
        Path to SHAP aggregation pickle file
    labels_csv_path : str
        Path to chapter labels CSV
    
    Returns
    -------
    dict
        Dictionary containing:
        - shap_data: raw SHAP data
        - id_to_token: token ID -> name mapping
        - token_to_chapter: token ID -> chapter mapping
        - chapter_to_color: chapter -> color mapping
        - token_counts: token frequency counts
    """
    print("="*60)
    print("Loading SHAP data...")
    print("="*60)
    
    with open(shap_pickle_path, 'rb') as f:
        shap_data = pickle.load(f)
    
    labels_df = pd.read_csv(labels_csv_path)
    
    id_to_token = labels_df.set_index('token_id')['name'].to_dict()
    token_to_chapter = labels_df.set_index('token_id')['Short Chapter'].to_dict()
    chapter_to_color = labels_df.groupby('Short Chapter')['color'].first().to_dict()
    
    token_counts = Counter(shap_data['tokens'])
    
    print(f"Total tokens: {len(shap_data['tokens'])}")
    print(f"SHAP values shape: {shap_data['values'].shape}")
    
    return {
        'shap_data': shap_data,
        'id_to_token': id_to_token,
        'token_to_chapter': token_to_chapter,
        'chapter_to_color': chapter_to_color,
        'token_counts': token_counts
    }

def process_shap_matrix(shap_info, exclude_chapters=None, min_occurrences=5, 
                        death_token_id=1288, death_repeat=10):
    """
    Process SHAP data into matrices for visualization
    
    Parameters
    ----------
    shap_info : dict
        Output from load_shap_data()
    exclude_chapters : list, optional
        Chapters to exclude
    min_occurrences : int
        Minimum token occurrences to include
    death_token_id : int
        Token ID for death
    death_repeat : int
        How many times to expand death column
    
    Returns
    -------
    dict
        Processed matrices and metadata
    """
    if exclude_chapters is None:
        exclude_chapters = ['Sex', 'Smoking, Alcohol and BMI', 'Technical']
    
    shap_data = shap_info['shap_data']
    token_counts = shap_info['token_counts']
    token_to_chapter = shap_info['token_to_chapter']
    
    print("\nProcessing SHAP matrix...")
    
    # Filter frequent tokens
    frequent_tokens = {}
    for token, count in token_counts.items():
        if count > min_occurrences:
            chapter = token_to_chapter.get(token, 'Unknown')
            if chapter not in exclude_chapters:
                frequent_tokens[token] = count
    
    # Add death token if missing
    if death_token_id not in frequent_tokens and death_token_id in token_counts:
        frequent_tokens[death_token_id] = token_counts[death_token_id]
        print(f"✓ Death token added (count: {frequent_tokens[death_token_id]})")
    
    # Group by chapter
    chapter_to_tokens = defaultdict(list)
    for token in frequent_tokens.keys():
        chapter = token_to_chapter.get(token, 'Unknown')
        if chapter != 'Unknown':
            chapter_to_tokens[chapter].append(token)
    
    # Sort chapters by Roman numeral
    def roman_to_int(roman_str):
        parts = roman_str.split('.')
        if len(parts) < 2:
            return 999
        roman = parts[0].strip()
        roman_map = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
            'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
            'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
            'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
            'XXI': 21
        }
        return roman_map.get(roman, 999)
    
    all_chapters = [ch for ch in chapter_to_tokens.keys() if ch not in exclude_chapters]
    chapter_order = sorted(all_chapters, key=roman_to_int)
    
    # Compute SHAP effects
    print("Computing SHAP effects...")
    frequent_token_set = set(frequent_tokens.keys())
    frequent_token_list = sorted(list(frequent_token_set))
    
    shap_effects = defaultdict(lambda: defaultdict(list))
    
    for idx in tqdm(range(len(shap_data['tokens']))):
        input_token = shap_data['tokens'][idx]
        if input_token not in frequent_token_set:
            continue
        
        shap_vals = shap_data['values'][idx]
        
        for predicted_token in frequent_token_list:
            if predicted_token < len(shap_vals):
                shap_value = shap_vals[predicted_token]
                if abs(shap_value) > 1e-6:
                    shap_effects[input_token][predicted_token].append(shap_value)
    
    # Build matrices
    present_tokens = [t for t in frequent_token_list if t != death_token_id]
    predicted_tokens = list(frequent_token_list)
    if death_token_id not in predicted_tokens:
        predicted_tokens.append(death_token_id)
    
    n_present = len(present_tokens)
    n_predicted = len(predicted_tokens)
    
    shap_matrix = np.zeros((n_present, n_predicted), dtype=np.float32)
    
    for i, input_token in enumerate(present_tokens):
        for j, predicted_token in enumerate(predicted_tokens):
            if predicted_token in shap_effects[input_token]:
                values = shap_effects[input_token][predicted_token]
                shap_matrix[i, j] = np.mean(values)
    
    return {
        'shap_matrix': shap_matrix,
        'present_tokens': present_tokens,
        'predicted_tokens': predicted_tokens,
        'chapter_order': chapter_order,
        'chapter_to_tokens': chapter_to_tokens,
        'frequent_tokens': frequent_tokens,
        'death_token_id': death_token_id,
        'death_repeat': death_repeat
    }

# =============================================================================
# Helper Functions
# =============================================================================

def extract_roman(chapter_str):
    """Extract Roman numeral from chapter name"""
    parts = chapter_str.split('.')
    if len(parts) >= 1:
        return parts[0].strip()
    return chapter_str

def apply_log_transform(matrix, strength=100):
    """Apply log transformation to SHAP values"""
    return np.sign(matrix) * np.log10(1 + strength * np.abs(matrix))

# =============================================================================
# Visualization Functions
# =============================================================================

def draw_full_heatmap(shap_info, matrix_info, use_log=False):
    """
    Draw full interaction heatmap with Death column expanded
    
    Parameters
    ----------
    shap_info : dict
        From load_shap_data()
    matrix_info : dict
        From process_shap_matrix()
    use_log : bool
        Whether to use log scale
    """
    print(f"\nCreating Full Interaction Heatmap {'(log scale)' if use_log else ''}...")
    
    shap_matrix = matrix_info['shap_matrix']
    present_tokens = matrix_info['present_tokens']
    predicted_tokens = matrix_info['predicted_tokens']
    chapter_order = matrix_info['chapter_order']
    chapter_to_tokens = matrix_info['chapter_to_tokens']
    death_token_id = matrix_info['death_token_id']
    death_repeat = matrix_info['death_repeat']
    
    id_to_token = shap_info['id_to_token']
    chapter_to_color = shap_info['chapter_to_color']
    
    # Order tokens by chapter
    ordered_present_tokens = []
    present_chapter_boundaries = [0]
    for chapter in chapter_order:
        tokens_in_chapter = sorted([t for t in chapter_to_tokens[chapter] if t != death_token_id])
        ordered_present_tokens.extend(tokens_in_chapter)
        present_chapter_boundaries.append(len(ordered_present_tokens))
    
    ordered_predicted_tokens = []
    predicted_chapter_boundaries = [0]
    for chapter in chapter_order:
        tokens_in_chapter = sorted(chapter_to_tokens[chapter])
        ordered_predicted_tokens.extend(tokens_in_chapter)
        predicted_chapter_boundaries.append(len(ordered_predicted_tokens))
    
    present_to_idx = {token: i for i, token in enumerate(present_tokens)}
    predicted_to_idx = {token: i for i, token in enumerate(predicted_tokens)}
    
    ordered_present_indices = [present_to_idx[token] for token in ordered_present_tokens if token in present_to_idx]
    ordered_predicted_indices = [predicted_to_idx[token] for token in ordered_predicted_tokens if token in predicted_to_idx]
    
    shap_matrix_ordered = shap_matrix[np.ix_(ordered_present_indices, ordered_predicted_indices)]
    
    # Expand Death column
    print(f"Expanding Death column: {death_repeat}x")
    
    expanded_columns = []
    token_positions = []
    cumulative_pos = 0
    
    for i, token in enumerate(ordered_predicted_tokens):
        col_data = shap_matrix_ordered[:, i:i+1]
        repeat_count = death_repeat if token == death_token_id else 1
        expanded = np.repeat(col_data, repeat_count, axis=1)
        expanded_columns.append(expanded)
        
        start_pos = cumulative_pos
        end_pos = cumulative_pos + repeat_count
        token_positions.append((start_pos, end_pos))
        cumulative_pos = end_pos
    
    shap_matrix_wide = np.concatenate(expanded_columns, axis=1)
    
    print(f"Original: {shap_matrix_ordered.shape} → Expanded: {shap_matrix_wide.shape}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(28, 20))
    
    # Apply log transform if needed
    if use_log:
        shap_display = apply_log_transform(shap_matrix_wide)
        vmax = np.percentile(np.abs(shap_display), 95)
    else:
        shap_display = shap_matrix_wide
        vmax = np.percentile(np.abs(shap_display), 95)
    
    im = ax.imshow(shap_display, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    # Add chapter backgrounds
    for i, chapter in enumerate(chapter_order):
        color = chapter_to_color.get(chapter, '#000000')
        
        # Y-axis background
        y_start = present_chapter_boundaries[i]
        y_end = present_chapter_boundaries[i+1]
        if y_end > y_start:
            rect_y = plt.Rectangle((-0.02 * shap_matrix_wide.shape[1], y_start-0.5), 
                                   0.02 * shap_matrix_wide.shape[1], 
                                   y_end - y_start,
                                   facecolor=color, alpha=0.7, 
                                   transform=ax.transData, clip_on=False)
            ax.add_patch(rect_y)
        
        # X-axis background
        x_start_orig = predicted_chapter_boundaries[i]
        x_end_orig = predicted_chapter_boundaries[i+1]
        
        if x_end_orig > x_start_orig:
            chapter_start_expanded = token_positions[x_start_orig][0]
            chapter_end_expanded = token_positions[x_end_orig - 1][1]
            
            rect_x = plt.Rectangle((chapter_start_expanded-0.5, -0.02 * len(ordered_present_tokens)), 
                                   chapter_end_expanded - chapter_start_expanded,
                                   0.02 * len(ordered_present_tokens),
                                   facecolor=color, alpha=0.7,
                                   transform=ax.transData, clip_on=False)
            ax.add_patch(rect_x)
    
    # Axis labels (Roman numerals + Death)
    chapter_labels_y, chapter_positions_y = [], []
    for i, chapter in enumerate(chapter_order):
        start, end = present_chapter_boundaries[i], present_chapter_boundaries[i+1]
        if end > start:
            chapter_positions_y.append((start + end) / 2)
            chapter_labels_y.append(extract_roman(chapter))
    
    chapter_labels_x, chapter_positions_x = [], []
    for i, chapter in enumerate(chapter_order):
        start_orig, end_orig = predicted_chapter_boundaries[i], predicted_chapter_boundaries[i+1]
        if end_orig > start_orig:
            chapter_start_expanded = token_positions[start_orig][0]
            chapter_end_expanded = token_positions[end_orig - 1][1]
            center = (chapter_start_expanded + chapter_end_expanded) / 2
            chapter_positions_x.append(center)
            
            is_death_chapter = any(token == death_token_id for token in chapter_to_tokens[chapter])
            chapter_labels_x.append('Death' if is_death_chapter else extract_roman(chapter))
    
    ax.set_yticks(chapter_positions_y)
    ax.set_yticklabels(chapter_labels_y, fontsize=10)
    ax.tick_params(axis='y', pad=12)
    ax.set_xticks(chapter_positions_x)
    ax.set_xticklabels(chapter_labels_x, fontsize=10, rotation=90)
    
    # Color labels
    for label, chapter in zip(ax.get_yticklabels(), chapter_order):
        if present_chapter_boundaries[chapter_order.index(chapter)+1] > present_chapter_boundaries[chapter_order.index(chapter)]:
            color = chapter_to_color.get(chapter, '#000000')
            label.set_color(color)
            label.set_fontweight('bold')
    
    for label, chapter in zip(ax.get_xticklabels(), chapter_order):
        if predicted_chapter_boundaries[chapter_order.index(chapter)+1] > predicted_chapter_boundaries[chapter_order.index(chapter)]:
            color = chapter_to_color.get(chapter, '#000000')
            label.set_color(color)
            label.set_fontweight('bold')
    
    title_suffix = ' (log scale)' if use_log else ''
    ax.set_title(f'Disease Interaction Heatmap{title_suffix}', fontsize=14, pad=20)
    ax.set_ylabel('Present token', fontsize=12)
    ax.set_xlabel('Predicted token', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = 'log(SHAP value)' if use_log else 'SHAP value'
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.show()  # Display inline
    
    return fig

def draw_focused_heatmap(shap_info, matrix_info, top_n=10, use_log=False):
    """
    Draw focused heatmap with top N tokens per chapter
    
    Parameters
    ----------
    shap_info : dict
        From load_shap_data()
    matrix_info : dict
        From process_shap_matrix()
    top_n : int
        Number of top tokens per chapter
    use_log : bool
        Whether to use log scale
    """
    print(f"\nCreating Focused Heatmap (Top {top_n}) {'(log scale)' if use_log else ''}...")
    
    shap_matrix = matrix_info['shap_matrix']
    present_tokens = matrix_info['present_tokens']
    predicted_tokens = matrix_info['predicted_tokens']
    chapter_order = matrix_info['chapter_order']
    chapter_to_tokens = matrix_info['chapter_to_tokens']
    frequent_tokens = matrix_info['frequent_tokens']
    death_token_id = matrix_info['death_token_id']
    death_repeat = matrix_info['death_repeat']
    
    id_to_token = shap_info['id_to_token']
    chapter_to_color = shap_info['chapter_to_color']
    
    # Select top N tokens per chapter
    selected_present_tokens = []
    selected_present_boundaries = [0]
    for chapter in chapter_order:
        tokens_in_chapter = [t for t in chapter_to_tokens[chapter] if t != death_token_id]
        tokens_sorted = sorted(tokens_in_chapter, 
                              key=lambda t: frequent_tokens.get(t, 0), reverse=True)
        selected_present_tokens.extend(tokens_sorted[:top_n])
        selected_present_boundaries.append(len(selected_present_tokens))
    
    selected_predicted_tokens = []
    selected_predicted_boundaries = [0]
    for chapter in chapter_order:
        tokens_sorted = sorted(chapter_to_tokens[chapter], 
                              key=lambda t: frequent_tokens.get(t, 0), reverse=True)
        selected_predicted_tokens.extend(tokens_sorted[:top_n])
        selected_predicted_boundaries.append(len(selected_predicted_tokens))
    
    present_to_idx = {token: i for i, token in enumerate(present_tokens)}
    predicted_to_idx = {token: i for i, token in enumerate(predicted_tokens)}
    
    selected_present_indices = [present_to_idx[token] for token in selected_present_tokens if token in present_to_idx]
    selected_predicted_indices = [predicted_to_idx[token] for token in selected_predicted_tokens if token in predicted_to_idx]
    
    shap_matrix_focused = shap_matrix[np.ix_(selected_present_indices, selected_predicted_indices)]
    
    # Expand Death column
    expanded_columns = []
    token_positions = []
    cumulative_pos = 0
    
    for i, token in enumerate(selected_predicted_tokens):
        col_data = shap_matrix_focused[:, i:i+1]
        repeat_count = death_repeat if token == death_token_id else 1
        expanded = np.repeat(col_data, repeat_count, axis=1)
        expanded_columns.append(expanded)
        
        start_pos = cumulative_pos
        end_pos = cumulative_pos + repeat_count
        token_positions.append((start_pos, end_pos))
        cumulative_pos = end_pos
    
    shap_matrix_wide = np.concatenate(expanded_columns, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    if use_log:
        shap_display = apply_log_transform(shap_matrix_wide)
        vmax = np.percentile(np.abs(shap_display), 95)
    else:
        shap_display = shap_matrix_wide
        vmax = np.percentile(np.abs(shap_display), 95)
    
    im = ax.imshow(shap_display, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    # Add chapter backgrounds (similar to full heatmap)
    for i, chapter in enumerate(chapter_order):
        color = chapter_to_color.get(chapter, '#000000')
        
        y_start = selected_present_boundaries[i]
        y_end = selected_present_boundaries[i+1]
        if y_end > y_start:
            rect_y = plt.Rectangle((-0.02 * shap_matrix_wide.shape[1], y_start-0.5), 
                                   0.02 * shap_matrix_wide.shape[1], 
                                   y_end - y_start,
                                   facecolor=color, alpha=0.7, 
                                   transform=ax.transData, clip_on=False)
            ax.add_patch(rect_y)
        
        x_start_orig = selected_predicted_boundaries[i]
        x_end_orig = selected_predicted_boundaries[i+1]
        
        if x_end_orig > x_start_orig:
            chapter_start_expanded = token_positions[x_start_orig][0]
            chapter_end_expanded = token_positions[x_end_orig - 1][1]
            
            rect_x = plt.Rectangle((chapter_start_expanded-0.5, -0.02 * len(selected_present_tokens)), 
                                   chapter_end_expanded - chapter_start_expanded,
                                   0.02 * len(selected_present_tokens),
                                   facecolor=color, alpha=0.7,
                                   transform=ax.transData, clip_on=False)
            ax.add_patch(rect_x)
    
    # Axis labels
    chapter_positions_y, chapter_labels_y = [], []
    for i, chapter in enumerate(chapter_order):
        start, end = selected_present_boundaries[i], selected_present_boundaries[i+1]
        if end > start:
            chapter_positions_y.append((start + end) / 2)
            chapter_labels_y.append(extract_roman(chapter))
    
    chapter_positions_x, chapter_labels_x = [], []
    for i, chapter in enumerate(chapter_order):
        start_orig, end_orig = selected_predicted_boundaries[i], selected_predicted_boundaries[i+1]
        if end_orig > start_orig:
            chapter_start_expanded = token_positions[start_orig][0]
            chapter_end_expanded = token_positions[end_orig - 1][1]
            center = (chapter_start_expanded + chapter_end_expanded) / 2
            chapter_positions_x.append(center)
            
            is_death_chapter = any(token == death_token_id for token in chapter_to_tokens[chapter])
            chapter_labels_x.append('Death' if is_death_chapter else extract_roman(chapter))
    
    ax.set_yticks(chapter_positions_y)
    ax.set_yticklabels(chapter_labels_y, fontsize=11)
    ax.set_xticks(chapter_positions_x)
    ax.set_xticklabels(chapter_labels_x, fontsize=11, rotation=90)
    
    # Color labels
    for label, chapter in zip(ax.get_yticklabels(), 
                             [ch for ch in chapter_order if selected_present_boundaries[chapter_order.index(ch)+1] > selected_present_boundaries[chapter_order.index(ch)]]):
        color = chapter_to_color.get(chapter, '#000000')
        label.set_color(color)
        label.set_fontweight('bold')
    
    for label, chapter in zip(ax.get_xticklabels(),
                             [ch for ch in chapter_order if selected_predicted_boundaries[chapter_order.index(ch)+1] > selected_predicted_boundaries[chapter_order.index(ch)]]):
        color = chapter_to_color.get(chapter, '#000000')
        label.set_color(color)
        label.set_fontweight('bold')
    
    title_suffix = ' (log scale)' if use_log else ''
    ax.set_title(f'Top {top_n} Diseases per Chapter (Death {death_repeat}x){title_suffix}', 
                 fontsize=14, pad=20)
    ax.set_ylabel('Present token (death excluded)', fontsize=12)
    ax.set_xlabel(f'Predicted token (death {death_repeat}x)', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = 'log(SHAP value)' if use_log else 'SHAP value'
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def draw_top_interactions(shap_info, matrix_info, top_n=20, use_log=False):
    """
    Draw bar plot of top disease interactions
    
    Parameters
    ----------
    shap_info : dict
        From load_shap_data()
    matrix_info : dict
        From process_shap_matrix()
    top_n : int
        Number of top interactions to show
    use_log : bool
        Whether to use log scale
    """
    print(f"\nCreating Top {top_n} Interactions {'(log scale)' if use_log else ''}...")
    
    shap_matrix = matrix_info['shap_matrix']
    present_tokens = matrix_info['present_tokens']
    predicted_tokens = matrix_info['predicted_tokens']
    id_to_token = shap_info['id_to_token']
    
    # Collect all interactions
    interactions = []
    for i, present_token in enumerate(present_tokens):
        for j, predicted_token in enumerate(predicted_tokens):
            if present_token != predicted_token:
                val = shap_matrix[i, j]
                if abs(val) > 0.01:
                    interactions.append({
                        'present': id_to_token.get(present_token, f"Token_{present_token}"),
                        'predicted': id_to_token.get(predicted_token, f"Token_{predicted_token}"),
                        'shap': val,
                        'abs_shap': abs(val)
                    })
    
    interactions_df = pd.DataFrame(interactions)
    top_interactions = interactions_df.nlargest(top_n, 'abs_shap')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if use_log:
        shap_display = apply_log_transform(top_interactions['shap'].values)
    else:
        shap_display = top_interactions['shap'].values
    
    colors = ['red' if x > 0 else 'blue' for x in shap_display]
    labels = [f"{row['present']} → {row['predicted']}" for _, row in top_interactions.iterrows()]
    
    ax.barh(range(len(top_interactions)), shap_display, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_interactions)))
    ax.set_yticklabels(labels, fontsize=8)
    xlabel = 'log(SHAP Value)' if use_log else 'SHAP Value'
    ax.set_xlabel(xlabel, fontsize=12)
    title_suffix = ' (log scale)' if use_log else ''
    ax.set_title(f'Top {top_n} Disease Interactions{title_suffix}', fontsize=14, pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def draw_chapter_to_chapter(shap_info, matrix_info, use_log=False, color_mode='linear'):
    """
    Draw chapter-to-chapter interaction heatmap
    
    Parameters
    ----------
    shap_info : dict
        From load_shap_data()
    matrix_info : dict
        From process_shap_matrix()
    use_log : bool
        Whether to use log scale
    color_mode : str
        Color enhancement mode:
        - 'power': Power normalization (gamma=0.5, emphasizes small values)
        - 'strong': Stronger power (gamma=0.3, more contrast)
        - 'linear': Standard linear scale
        - 'percentile': Use narrow percentile range (stronger contrast)
    """
    print(f"\nCreating Chapter-to-Chapter Heatmap {'(log scale)' if use_log else ''} [color: {color_mode}]...")
    
    # This requires the ordered matrix - compute it here
    shap_matrix = matrix_info['shap_matrix']
    present_tokens = matrix_info['present_tokens']
    predicted_tokens = matrix_info['predicted_tokens']
    chapter_order = matrix_info['chapter_order']
    chapter_to_tokens = matrix_info['chapter_to_tokens']
    death_token_id = matrix_info['death_token_id']
    chapter_to_color = shap_info['chapter_to_color']
    
    # Order tokens by chapter
    ordered_present_tokens = []
    present_chapter_boundaries = [0]
    for chapter in chapter_order:
        tokens_in_chapter = sorted([t for t in chapter_to_tokens[chapter] if t != death_token_id])
        ordered_present_tokens.extend(tokens_in_chapter)
        present_chapter_boundaries.append(len(ordered_present_tokens))
    
    ordered_predicted_tokens = []
    predicted_chapter_boundaries = [0]
    for chapter in chapter_order:
        tokens_in_chapter = sorted(chapter_to_tokens[chapter])
        ordered_predicted_tokens.extend(tokens_in_chapter)
        predicted_chapter_boundaries.append(len(ordered_predicted_tokens))
    
    present_to_idx = {token: i for i, token in enumerate(present_tokens)}
    predicted_to_idx = {token: i for i, token in enumerate(predicted_tokens)}
    
    ordered_present_indices = [present_to_idx[token] for token in ordered_present_tokens if token in present_to_idx]
    ordered_predicted_indices = [predicted_to_idx[token] for token in ordered_predicted_tokens if token in predicted_to_idx]
    
    shap_matrix_ordered = shap_matrix[np.ix_(ordered_present_indices, ordered_predicted_indices)]
    
    # Compute chapter-level aggregation
    n_chapters = len(chapter_order)
    chapter_matrix = np.zeros((n_chapters, n_chapters))
    
    for i, ch_present in enumerate(chapter_order):
        for j, ch_predicted in enumerate(chapter_order):
            start_i, end_i = present_chapter_boundaries[i], present_chapter_boundaries[i+1]
            start_j, end_j = predicted_chapter_boundaries[j], predicted_chapter_boundaries[j+1]
            
            if end_i > start_i and end_j > start_j:
                block = shap_matrix_ordered[start_i:end_i, start_j:end_j]
                chapter_matrix[i, j] = np.mean(block)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if use_log:
        chapter_display = apply_log_transform(chapter_matrix)
    else:
        chapter_display = chapter_matrix
    
    # Select color mapping strategy
    from matplotlib.colors import PowerNorm, Normalize
    
    abs_values = np.abs(chapter_display)
    
    if color_mode == 'strong':
        # High contrast scaling (gamma=0.3)
        vmax = np.percentile(abs_values[abs_values > 0], 85)
        norm = PowerNorm(gamma=0.3, vmin=-vmax, vmax=vmax)
        print(f"  Using STRONG power norm (gamma=0.3, vmax={vmax:.4f})")
        
    elif color_mode == 'power':
        # 중간 대비 (gamma=0.5, 기본값)
        vmax = np.percentile(abs_values[abs_values > 0], 90)
        norm = PowerNorm(gamma=0.5, vmin=-vmax, vmax=vmax)
        print(f"  Using power norm (gamma=0.5, vmax={vmax:.4f})")
        
    elif color_mode == 'percentile':
        # Clip extreme values using a narrower percentile range
        vmax = np.percentile(abs_values[abs_values > 0], 80)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        print(f"  Using percentile norm (80th, vmax={vmax:.4f})")
        
    else:  # 'linear'
        # Standard linear scaling
        vmax = np.percentile(abs_values, 95)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        print(f"  Using linear norm (vmax={vmax:.4f})")
    
    im = ax.imshow(chapter_display, cmap='RdBu_r', aspect='auto',
                   norm=norm, interpolation='nearest')
    
    # Labels
    roman_labels_y = [extract_roman(ch) for ch in chapter_order]
    roman_labels_x = ['Death' if any(token == death_token_id for token in chapter_to_tokens[ch]) 
                      else extract_roman(ch) for ch in chapter_order]
    
    ax.set_xticks(range(n_chapters))
    ax.set_yticks(range(n_chapters))
    ax.set_xticklabels(roman_labels_x, rotation=90, fontsize=10)
    ax.set_yticklabels(roman_labels_y, fontsize=10)
    
    for i, chapter in enumerate(chapter_order):
        color = chapter_to_color.get(chapter, '#000000')
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight('bold')
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_fontweight('bold')
    
    title_suffix = ' (log scale)' if use_log else ''
    ax.set_title(f'Chapter-to-Chapter Interactions{title_suffix}', fontsize=14, pad=20)
    ax.set_ylabel('Present Chapter', fontsize=12)
    ax.set_xlabel('Predicted Chapter', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = 'Mean log(SHAP value)' if use_log else 'Mean SHAP value'
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def draw_death_risk_factors(shap_info, matrix_info, top_n=10):
    """
    Draw death risk factors (increasing and decreasing)
    
    Parameters
    ----------
    shap_info : dict
        From load_shap_data()
    matrix_info : dict
        From process_shap_matrix()
    top_n : int
        Number of top factors to show on each side
    """
    print(f"\nCreating Death Risk Factors (Top {top_n})...")
    
    shap_matrix = matrix_info['shap_matrix']
    present_tokens = matrix_info['present_tokens']
    predicted_tokens = matrix_info['predicted_tokens']
    death_token_id = matrix_info['death_token_id']
    
    id_to_token = shap_info['id_to_token']
    token_to_chapter = shap_info['token_to_chapter']
    chapter_to_color = shap_info['chapter_to_color']
    
    if death_token_id not in predicted_tokens:
        print("⚠ Death token not found")
        return None
    
    death_col_idx = predicted_tokens.index(death_token_id)
    
    # Collect all death effects
    death_effects = []
    for i, token in enumerate(present_tokens):
        death_shap = shap_matrix[i, death_col_idx]
        death_effects.append({
            'disease': id_to_token.get(token, f"Token_{token}"),
            'chapter': token_to_chapter.get(token, 'Unknown'),
            'shap_value': death_shap,
            'abs_shap': abs(death_shap)
        })
    
    death_df = pd.DataFrame(death_effects)
    death_df = death_df.sort_values('shap_value', ascending=False)
    
    print(f"  Total diseases: {len(death_df)}")
    print(f"  Non-zero SHAP values: {(death_df['shap_value'] != 0).sum()}")
    print(f"  SHAP range: [{death_df['shap_value'].min():.6f}, {death_df['shap_value'].max():.6f}]")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Top N increasing
    top_increase = death_df.head(top_n)
    colors_increase = [chapter_to_color.get(row['chapter'], 'gray') for _, row in top_increase.iterrows()]
    ax1.barh(range(len(top_increase)), top_increase['shap_value'], color=colors_increase, alpha=0.7)
    ax1.set_yticks(range(len(top_increase)))
    ax1.set_yticklabels(top_increase['disease'], fontsize=7)
    ax1.set_xlabel('SHAP Value', fontsize=11)
    ax1.set_title(f'Top {top_n} Diseases Increasing Death Risk', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Top N decreasing
    top_decrease = death_df.tail(top_n).iloc[::-1]
    colors_decrease = [chapter_to_color.get(row['chapter'], 'gray') for _, row in top_decrease.iterrows()]
    ax2.barh(range(len(top_decrease)), top_decrease['shap_value'], color=colors_decrease, alpha=0.7)
    ax2.set_yticks(range(len(top_decrease)))
    ax2.set_yticklabels(top_decrease['disease'], fontsize=7)
    ax2.set_xlabel('SHAP Value', fontsize=11)
    ax2.set_title(f'Top {top_n} Diseases Decreasing Death Risk', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"  Top increaser: {top_increase.iloc[0]['disease']} (+{top_increase.iloc[0]['shap_value']:.6f})")
    print(f"  Top decreaser: {top_decrease.iloc[0]['disease']} ({top_decrease.iloc[0]['shap_value']:.6f})")
    
    return fig

def draw_shap_distribution(matrix_info):
    """
    Draw distribution of SHAP values
    
    Parameters
    ----------
    matrix_info : dict
        From process_shap_matrix()
    """
    print("\nCreating SHAP Distribution...")
    
    shap_matrix = matrix_info['shap_matrix']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_shap_values = shap_matrix.flatten()
    all_shap_values = all_shap_values[all_shap_values != 0]
    
    ax.hist(all_shap_values, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('SHAP Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of SHAP Values', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig