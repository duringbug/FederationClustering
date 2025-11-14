from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os

# -------------------------------
# 1️⃣ 清洗文本函数
# -------------------------------
def clean_wiki40b_text(text):
    """清洗单条 Wiki-40B 文本"""
    clean_text = re.sub(r"_START_.*?_|_NEWLINE_", " ", text)
    return clean_text.strip()

# -------------------------------
# 2️⃣ 初始化模型
# -------------------------------
model = SentenceTransformer("intfloat/multilingual-e5-base")
batch_size = 64  # 可根据显存调整

# -------------------------------
# 3️⃣ 输出目录
# -------------------------------
output_dir = "wiki40b_vectors_5000"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 4️⃣ 加载本地缓存数据 + 只取前 5000 条
# -------------------------------
ds = load_dataset("google/wiki40b", "en", split="train")
ds_small = ds.select(range(5000))  # 只取前 5000 条

# -------------------------------
# 5️⃣ 批量生成向量并写 Parquet
# -------------------------------
all_rows = []
count = 0
pbar = tqdm(total=len(ds_small), desc="Processing")

batch_texts = []
for item in ds_small:
    text = clean_wiki40b_text(item["text"])
    batch_texts.append(text)

    if len(batch_texts) >= batch_size:
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        for t, emb in zip(batch_texts, embeddings):
            all_rows.append({"text": t, "embedding": emb.tolist()})
            count += 1

        batch_texts = []
        pbar.update(batch_size)

        # 每 1000 条写入一次 Parquet
        if len(all_rows) >= 1000:
            table = pa.Table.from_pylist(all_rows)
            pq.write_to_dataset(table, root_path=output_dir, existing_data_behavior="overwrite_or_ignore")
            all_rows = []

# 写入剩余数据
if batch_texts:
    embeddings = model.encode(batch_texts, show_progress_bar=False)
    for t, emb in zip(batch_texts, embeddings):
        all_rows.append({"text": t, "embedding": emb.tolist()})
        count += 1
    pbar.update(len(batch_texts))

if all_rows:
    table = pa.Table.from_pylist(all_rows)
    pq.write_to_dataset(table, root_path=output_dir, existing_data_behavior="overwrite_or_ignore")

pbar.close()
print(f"Saved {count} vectors to {output_dir}/")
