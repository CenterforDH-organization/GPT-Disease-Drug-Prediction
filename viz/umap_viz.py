import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import umap
import os
from collections import Counter
from model import CompositeDelphi, CompositeDelphiConfig

# =============================================================================
# 0. Global configuration
# =============================================================================
CKPT_PATH = "CKPT.pt"
LABELS_CHAPTER_PATH = "labels_chapter"
DATA_BIN_PATH = "data.bin"
OUTPUT_FILE = "umap.png"

START_TOKEN_ID = 23
TARGET_LABEL_IDS = [23, 1288] # 이름 표시할 토큰

# 크기 설정 (3단계)
SIZE_LEVELS = {
    'Low': {'thresh': 1000, 'size': 30, 'label': '< 1k'},
    'Mid': {'thresh': 50000, 'size': 100, 'label': '1k ~ 50k'},
    'High': {'thresh': float('inf'), 'size': 350, 'label': '> 50k'}
}
DEFAULT_SIZE = 30 # 데이터 파일 없을 때 기본 크기

# UMAP 파라미터
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'

# =============================================================================
# 1. Token frequency analysis
# =============================================================================
print(f"[INFO] Computing token frequencies from dataset: {DATA_BIN_PATH}")
token_counts = {}

if os.path.exists(DATA_BIN_PATH):
    try:
        # Composite 데이터 구조 정의 (ID, AGE, DATA, SHIFT, TOTAL)
        composite_dtype = np.dtype([
            ('ID', np.uint32), ('AGE', np.uint32), ('DATA', np.uint32),
            ('SHIFT', np.uint32), ('TOTAL', np.uint32)
        ])
        
        # 파일 읽기
        data_raw = np.fromfile(DATA_BIN_PATH, dtype=composite_dtype)
        
        # DATA 컬럼만 추출하여 카운트
        # [중요] 데이터셋의 Raw Token 값에 +1을 해야 모델의 Token ID가 됨
        raw_tokens = data_raw['DATA']
        shifted_tokens = raw_tokens + 1
        
        # 카운팅 (numpy unique 사용이 빠름)
        unique, counts = np.unique(shifted_tokens, return_counts=True)
        token_counts = dict(zip(unique, counts))
        
        print(f"[OK] Frequency computation completed. Unique tokens: {len(token_counts)}")

        
        # (옵션) 상위 빈도 토큰 확인
        # top5 = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        # print(f"   Top 5 Freq: {top5}")
        
    except Exception as e:
        print(f"[WARN] Failed to load dataset ({e}). Falling back to default marker sizes.")
else:
    print("[WARN] Dataset file not found. Falling back to default marker sizes.")

# =============================================================================
# 2. Chapter metadata loading
# =============================================================================
print("[INFO] Loading chapter metadata...")
try:
    chapter_df = pd.read_csv(LABELS_CHAPTER_PATH, header=None)
    num_cols = chapter_df.shape[1]
    if num_cols == 6:
        chapter_df.columns = ['raw_idx', 'name', 'token_id', 'chapter_full', 'chapter_short', 'color']
    elif num_cols == 5:
        chapter_df.columns = ['name', 'token_id', 'chapter_full', 'chapter_short', 'color']
    else:
        raise ValueError("컬럼 수 오류")

    chapter_df['token_id'] = pd.to_numeric(chapter_df['token_id'], errors='coerce')
    chapter_df = chapter_df.dropna(subset=['token_id'])
    chapter_df['token_id'] = chapter_df['token_id'].astype(int)
    
    filtered_df = chapter_df[chapter_df['token_id'] >= START_TOKEN_ID].copy()
    token_meta = filtered_df.set_index('token_id').to_dict('index')
    
    # 챕터 범례 정보
    legend_info = filtered_df[['chapter_short', 'color']].drop_duplicates()
    color_patches = [mpatches.Patch(color=row['color'], label=row['chapter_short']) 
                     for _, row in legend_info.iterrows()]

except Exception as e:
    print(f"[ERROR] Failed to load chapter metadata: {e}")
    exit()

# =============================================================================
# 3. Model loading and embedding extraction
# =============================================================================
print("[INFO] Loading model checkpoint...")
device = "cpu"
try:
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    config = CompositeDelphiConfig(**checkpoint['model_args'])
    model = CompositeDelphi(config)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    full_embeddings = model.composite_emb.data_emb.weight.detach().numpy()
    valid_token_ids = sorted(token_meta.keys())
    filtered_embeddings = full_embeddings[valid_token_ids]

except Exception as e:
    print(f"[ERROR] Model initialization or embedding extraction failed: {e}")
    exit()

# =============================================================================
# 4. UMAP projection
# =============================================================================
print("[INFO] Running UMAP dimensionality reduction...")
reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=42)
embedding_2d = reducer.fit_transform(filtered_embeddings)

df_plot = pd.DataFrame(embedding_2d, columns=['x', 'y'])
df_plot['token_id'] = valid_token_ids
df_plot['color'] = [token_meta[tid]['color'] for tid in valid_token_ids]
df_plot['name'] = [token_meta[tid]['name'] for tid in valid_token_ids]

# [핵심] 빈도에 따른 크기 할당
sizes = []
for tid in valid_token_ids:
    count = token_counts.get(tid, 0)
    if count < SIZE_LEVELS['Low']['thresh']:
        sizes.append(SIZE_LEVELS['Low']['size'])
    elif count < SIZE_LEVELS['Mid']['thresh']:
        sizes.append(SIZE_LEVELS['Mid']['size'])
    else:
        sizes.append(SIZE_LEVELS['High']['size'])
df_plot['size'] = sizes

# =============================================================================
# 5. Visualization (size scaled by frequency)
# =============================================================================
print("[INFO] Rendering scatter plot...")
plt.figure(figsize=(18, 14)) # 범례 공간 확보를 위해 조금 더 크게

# Scatter Plot (크기 s 옵션 사용)
plt.scatter(
    df_plot['x'], 
    df_plot['y'], 
    c=df_plot['color'], 
    s=df_plot['size'],  # 빈도별 크기 적용
    alpha=0.7, 
    edgecolors='white', 
    linewidth=0.3
)

# 선택적 라벨링
if TARGET_LABEL_IDS:
    texts = []
    for i, row in df_plot.iterrows():
        tid = int(row['token_id'])
        if tid in TARGET_LABEL_IDS:
            label_text = f"{row['name']} ({tid})"
            t = plt.text(row['x'], row['y'], label_text, fontsize=10, fontweight='bold', color='black')
            texts.append(t)
    
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except: pass

# ---------------------------------------------------------
# [범례 1] 챕터 (색상)
# ---------------------------------------------------------
# 첫 번째 범례 추가
legend1 = plt.legend(
    handles=color_patches, 
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    title="Chapters",
    fontsize=9
)
plt.gca().add_artist(legend1) # 중요: 두 번째 범례를 위해 첫 번째를 artist로 고정

# ---------------------------------------------------------
# [범례 2] 빈도 (크기)
# ---------------------------------------------------------
# 크기 범례용 더미 점 생성
size_handles = []
for level in ['Low', 'Mid', 'High']:
    info = SIZE_LEVELS[level]
    # 회색 점으로 크기 예시
    handle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='gray',
                           markersize=np.sqrt(info['size']), # scatter의 s는 면적이므로 sqrt해서 마커사이즈로
                           label=info['label'])
    size_handles.append(handle)

plt.legend(
    handles=size_handles,
    bbox_to_anchor=(1.02, 0.4), # 색상 범례 아래쪽에 배치
    loc='upper left',
    title="Frequency",
    fontsize=10,
    labelspacing=1.5 # 점 사이 간격
)

plt.title("Tokens UMAP", fontsize=20)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"[OK] Output saved to disk: {OUTPUT_FILE}")