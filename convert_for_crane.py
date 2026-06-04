# convert_for_crane.py
"""
将旧 MMMU-BA 数据集格式转换为 CRANE 所需格式。
输出：
  - data/MOSI/label.csv        (video_id, clip_id, text, label, mode)
  - data/MOSI/wav/{video_id}/{clip_id}.wav   (音频重组)

前提：config.py 中的路径可正常访问旧数据集。
"""
import os
import sys
import pickle
import pandas as pd
import shutil
from tqdm import tqdm

# 导入旧 config 中的路径
from config import *
# ============================================================
# Step 1: 从 pkl 读取标签
# ============================================================
print("[1/5] Reading labels from pkl...")
with open(LABEL_PATH, 'rb') as f:
    data = pickle.load(f)

rows = []
for split in ['train', 'valid', 'test']:
    for i in range(len(data[split]['id'])):
        id_info = data[split]['id'][i]
        label_val = data[split]['labels'][i]
        segment_id = id_info[0].decode('utf-8')
        sentiment = float(label_val[0][0])
        rows.append({
            'split': split,
            'segment_id': segment_id,
            'sentiment': sentiment
        })

df = pd.DataFrame(rows)
print(f"   Loaded {len(df)} samples")

# ============================================================
# Step 2: 拆分 segment_id → video_id + clip_id
#         如 "03bSnISJlfM_1" → video_id="03bSnISJlfM", clip_id="1"
# ============================================================
print("[2/5] Parsing video_id / clip_id from segment_id...")
# 用 rsplit 应对 video_id 可能包含下划线的情况
df['video_id'] = df['segment_id'].apply(lambda x: x.rsplit('_', 1)[0])
df['clip_id']  = df['segment_id'].apply(lambda x: x.rsplit('_', 1)[1])
print(f"   Unique videos: {df['video_id'].nunique()}")

# ============================================================
# Step 3: 从 .annotprocessed 文件提取文本
# ============================================================
print("[3/5] Extracting text from annotprocessed files...")
text_dict = {}
transcript_files = sorted([
    os.path.join(TRANSCRIPT_DIR, f)
    for f in os.listdir(TRANSCRIPT_DIR)
    if f.endswith('.annotprocessed')
])

for fpath in transcript_files:
    video_id = os.path.splitext(os.path.basename(fpath))[0]
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if '_DELIM_' in line:
            parts = line.split('_DELIM_', 1)
            seg_idx = parts[0].strip()
            text = parts[1].strip()
        else:
            seg_idx = str(i + 1)
            text = line
        seg_id = f"{video_id}_{seg_idx}"
        text_dict[seg_id] = text

df['text'] = df['segment_id'].map(text_dict).fillna('')
missing_texts = (df['text'] == '').sum()
if missing_texts > 0:
    print(f"   ⚠ {missing_texts} samples have no matching text (will be empty string)")

# ============================================================
# Step 4: 生成新 CSV
# ============================================================
print("[4/5] Generating label.csv...")
split_map = {'train': 'train', 'valid': 'valid', 'test': 'test'}
df['mode'] = df['split'].map(split_map)

output_df = df[['video_id', 'clip_id', 'text', 'sentiment', 'mode']].copy()
output_df = output_df.rename(columns={'sentiment': 'label'})

# 确保 clip_id 为字符串（避免后续 str() 转换出错）
output_df['clip_id'] = output_df['clip_id'].astype(str)

csv_dir = 'data/MOSI'
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, 'label.csv')
output_df.to_csv(csv_path, index=False)

# 统计
for mode in ['train', 'valid', 'test']:
    subset = output_df[output_df['mode'] == mode]
    print(f"   {mode}: {len(subset)} samples")

# ============================================================
# Step 5: 重组音频文件
#   from: {AUDIO_DIR}/{segment_id}.wav    (平铺)
#   to:   data/MOSI/wav/{video_id}/{clip_id}.wav
# ============================================================
print("[5/5] Reorganizing audio files...")
USE_SYMLINK = False   # True=符号链接(省空间), False=物理复制

audio_out = os.path.join(csv_dir, 'wav')
os.makedirs(audio_out, exist_ok=True)

copied, missing = 0, 0
for _, row in tqdm(output_df.iterrows(), total=len(output_df), desc="  Copying"):
    video_id = row['video_id']
    clip_id  = row['clip_id']
    segment_id = f"{video_id}_{clip_id}"

    src = os.path.join(AUDIO_DIR, f"{segment_id}.wav")
    dst_dir = os.path.join(audio_out, video_id)
    dst = os.path.join(dst_dir, f"{clip_id}.wav")

    if not os.path.exists(src):
        missing += 1
        continue

    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(dst):
        if USE_SYMLINK:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
    copied += 1

print(f"   Copied/symlinked: {copied}, Missing: {missing}")

# ============================================================
# 完成
# ============================================================
print(f"\n✅ Done!")
print(f"   CSV  → {os.path.abspath(csv_path)}")
print(f"   Audio → {os.path.abspath(audio_out)}")
print(f"\n   Now update data_loader.py with:")
print(f"     csv_path = '{os.path.abspath(csv_path)}'")
print(f"     audio_file_path = '{os.path.abspath(audio_out)}'")