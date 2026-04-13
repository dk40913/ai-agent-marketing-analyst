"""
Phase D 共用模組：資料載入 + Embedding

三個 DB（ChromaDB / Pinecone / pgvector）都用這份模組，確保：
- 同樣的文字資料
- 同樣的 embedding 模型（all-MiniLM-L6-v2，輸出 384 維向量）
- 同樣的查詢問句

為什麼用 sentence-transformers 而不是 Gemini Embedding？
因為 Gemini Embedding 是付費 API，sentence-transformers 是本機免費模型，
在比較 DB 效能時不需要引入外部 API 的延遲和成本。
"""

import os
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")

# all-MiniLM-L6-v2：輕量、快速、384 維，適合語意搜尋
# 第一次執行會自動下載模型（約 80MB），之後快取在本機
model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# 三個 DB 都用同樣的測試查詢
TEST_QUERIES = [
    "山嵐咖啡線上銷售下滑的原因是什麼？",
    "城市動力的廣告表現如何？",
    "鮮橙生活有哪些風險？",
]


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """把長文字切成有重疊的小段，避免語意被切斷。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]


def load_documents() -> list[dict]:
    """
    讀取知識庫所有 .txt 檔，切段後回傳。
    每筆資料格式：
      { "id": "doc_0", "text": "...", "source": "xx.txt", "category": "brand_profiles" }
    """
    docs = []
    doc_id = 0

    for subfolder in ["brand_profiles", "market_reports", "anomaly_logs"]:
        folder_path = os.path.join(KB_DIR, subfolder)
        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                docs.append({
                    "id": f"doc_{doc_id}",
                    "text": chunk,
                    "source": filename,
                    "category": subfolder,
                })
                doc_id += 1

    print(f"📄 載入 {len(docs)} 個段落")
    return docs


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    用 sentence-transformers 把文字轉成向量。
    回傳格式：List[List[float]]，每個向量有 384 個浮點數。
    """
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()
