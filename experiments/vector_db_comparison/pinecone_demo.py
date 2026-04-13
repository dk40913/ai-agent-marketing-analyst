"""
Phase D - Pinecone 示範

Pinecone 是雲端向量資料庫，資料存在 Pinecone 的伺服器。
特色：
  - 完全託管，不需要自己維護伺服器
  - 適合生產環境，支援大規模查詢
  - 有免費額度（1 個 index，最多 100 萬向量）
  - 需要 API Key，查詢會有網路延遲

注意：Pinecone 的 index 建立可能需要 30–60 秒，程式會等待它 Ready。
"""

import os
import sys
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# 讀取 .env 裡的 PINECONE_API_KEY
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

sys.path.insert(0, os.path.dirname(__file__))
from shared import load_documents, get_embeddings, TEST_QUERIES, EMBEDDING_DIM

INDEX_NAME = "marketing-analyst"


def setup():
    """建立 Pinecone index 並上傳向量。"""
    print("\n🟡 Pinecone 設定中...")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("找不到 PINECONE_API_KEY，請確認 .env 檔案")

    pc = Pinecone(api_key=api_key)

    # 如果 index 已存在就刪掉重建（確保資料乾淨）
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"  刪除舊 index: {INDEX_NAME}")
        pc.delete_index(INDEX_NAME)
        time.sleep(5)

    # 建立 Serverless index
    # dimension 必須和 embedding 模型輸出維度一致（all-MiniLM-L6-v2 = 384）
    # metric="cosine"：用餘弦相似度計算兩個向量的相似程度
    # ServerlessSpec：Pinecone 免費版使用 Serverless 方案，不需要預先分配機器
    print(f"  建立 index（dimension={EMBEDDING_DIM}, metric=cosine）...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # 等 index 變成 Ready 狀態才能寫入
    print("  等待 index Ready", end="", flush=True)
    for _ in range(30):
        status = pc.describe_index(INDEX_NAME).status
        if status.get("ready"):
            break
        print(".", end="", flush=True)
        time.sleep(3)
    print(" ✓")

    index = pc.Index(INDEX_NAME)
    docs = load_documents()

    # 產生 embedding 向量
    texts = [d["text"] for d in docs]
    vectors = get_embeddings(texts)

    # Pinecone 要求的格式：List of (id, vector, metadata)
    # 分批上傳（每批 100 筆），避免單次請求太大
    t_start = time.time()
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_vectors = vectors[i:i+batch_size]
        upsert_data = [
            (d["id"], v, {"source": d["source"], "category": d["category"], "text": d["text"]})
            for d, v in zip(batch, batch_vectors)
        ]
        index.upsert(vectors=upsert_data)

    elapsed = time.time() - t_start
    print(f"✅ Pinecone 寫入 {len(docs)} 筆，耗時 {elapsed:.2f}s")
    return index, elapsed


def query(index, question: str, n: int = 3) -> dict:
    """
    把問句轉成向量，在 Pinecone 找最相似的 n 筆。
    include_metadata=True：讓結果帶回 source、text 等欄位。
    """
    q_vector = get_embeddings([question])[0]

    t_start = time.time()
    results = index.query(
        vector=q_vector,
        top_k=n,
        include_metadata=True,
    )
    elapsed = time.time() - t_start

    matches = results["matches"]
    return {
        "db": "Pinecone",
        "question": question,
        "results": [m["metadata"].get("text", "") for m in matches],
        "sources": [m["metadata"].get("source", "") for m in matches],
        "scores": [round(m["score"], 4) for m in matches],
        "query_time": elapsed,
    }


if __name__ == "__main__":
    index, ingest_time = setup()

    print("\n🔍 查詢測試：")
    for q in TEST_QUERIES:
        r = query(index, q)
        print(f"\n問：{r['question']}")
        print(f"來源：{r['sources']}")
        print(f"相似度分數：{r['scores']}")
        print(f"查詢耗時：{r['query_time']*1000:.1f}ms")
        print(f"第一段結果：{r['results'][0][:100]}...")
