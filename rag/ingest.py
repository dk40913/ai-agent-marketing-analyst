"""
將知識庫文字檔切段後存入 ChromaDB。
只需執行一次，之後 query.py 就能搜尋。

流程：
  讀取 .txt 檔 → 切成小段 → embedding → 存入 ChromaDB
"""

import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb")

# ChromaDB 存在本地資料夾，不需要網路
client = chromadb.PersistentClient(path=CHROMA_DIR)

# 用 ChromaDB 內建的 embedding（sentence-transformers，不需要 API Key）
ef = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="marketing_knowledge",
    embedding_function=ef,
)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    把長文字切成有重疊的小段。
    overlap（重疊）是為了避免一個句子剛好被切在兩段的交界，導致語意不完整。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]  # 過濾掉太短的殘段


def load_and_ingest():
    docs, ids, metadatas = [], [], []
    doc_id = 0

    # 遍歷三個子目錄
    for subfolder in ["brand_profiles", "market_reports", "anomaly_logs"]:
        folder_path = os.path.join(KB_DIR, subfolder)
        for filename in os.listdir(folder_path):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            for chunk in chunks:
                docs.append(chunk)
                ids.append(f"doc_{doc_id}")
                metadatas.append({
                    "source": filename,
                    "category": subfolder,
                })
                doc_id += 1

    # 一次批次存入 ChromaDB
    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    print(f"✅ 已存入 {len(docs)} 個段落，來自 {doc_id} 份文件的切段")


if __name__ == "__main__":
    # 避免重複存入（重新執行時先清空）
    client.delete_collection("marketing_knowledge")
    collection = client.get_or_create_collection(
        name="marketing_knowledge",
        embedding_function=ef,
    )
    load_and_ingest()
