"""
Phase D - ChromaDB 示範

ChromaDB 是本機向量資料庫，資料存在磁碟資料夾裡。
特色：
  - 完全免費，不需要網路
  - 安裝簡單（pip install chromadb）
  - 適合開發階段和小型專案
  - 資料量大（百萬筆）時效能會下降

這裡重新建立一個獨立的 collection 供比較實驗用，
不影響 rag/ 資料夾裡的正式 ChromaDB。
"""

import time
from typing import cast
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings
from shared import load_documents, get_embeddings, TEST_QUERIES, EMBEDDING_DIM

# 實驗用的 ChromaDB 放在這個資料夾，不污染正式的
CHROMA_DIR = "./chroma_experiment"
COLLECTION_NAME = "phase_d_comparison"


class CustomEmbeddingFunction(EmbeddingFunction):
    """
    把 shared.py 的 get_embeddings 包裝成 ChromaDB 接受的格式。
    ChromaDB 要求 embedding function 必須繼承 EmbeddingFunction 類別。
    """
    def __call__(self, input: list[str]) -> Embeddings:
        return cast(Embeddings, get_embeddings(input))


def setup():
    """建立 ChromaDB collection 並存入資料。"""
    print("\n🔵 ChromaDB 設定中...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 每次重新執行都清空重建，確保資料一致
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = CustomEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    docs = load_documents()

    t_start = time.time()
    collection.add(
        documents=[d["text"] for d in docs],
        ids=[d["id"] for d in docs],
        metadatas=[{"source": d["source"], "category": d["category"]} for d in docs],
    )
    elapsed = time.time() - t_start

    print(f"✅ ChromaDB 寫入 {len(docs)} 筆，耗時 {elapsed:.2f}s")
    return collection, elapsed


def query(collection, question: str, n: int = 3) -> dict:
    """對 ChromaDB 做語意搜尋，回傳最相關的 n 筆結果。"""
    t_start = time.time()
    results = collection.query(query_texts=[question], n_results=n)
    elapsed = time.time() - t_start

    return {
        "db": "ChromaDB",
        "question": question,
        "results": results["documents"][0],
        "sources": [m["source"] for m in results["metadatas"][0]],
        "query_time": elapsed,
    }


if __name__ == "__main__":
    collection, ingest_time = setup()

    print("\n🔍 查詢測試：")
    for q in TEST_QUERIES:
        r = query(collection, q)
        print(f"\n問：{r['question']}")
        print(f"來源：{r['sources']}")
        print(f"查詢耗時：{r['query_time']*1000:.1f}ms")
        print(f"第一段結果：{r['results'][0][:100]}...")
