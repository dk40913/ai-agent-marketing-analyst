"""
Phase D - pgvector 示範

pgvector 是 PostgreSQL 的向量擴充，讓傳統關聯式資料庫也能做向量搜尋。
特色：
  - 向量和一般欄位（文字、數字、日期）存在同一張表
  - 可以混合 SQL 查詢和向量搜尋（例如：「找鮮橙生活的文件，且相似度 > 0.8」）
  - 適合已有 PostgreSQL 的專案，不需要另外維護一個向量 DB
  - 本機 Docker 執行，完全免費

執行前需要先確認 Docker 有啟動：colima start
pgvector 容器會在這個程式裡自動啟動（如果還沒跑的話）。
"""

import os
import sys
import time
import subprocess
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(__file__))
from shared import load_documents, get_embeddings, TEST_QUERIES, EMBEDDING_DIM

# PostgreSQL 連線設定（對應 docker run 指令裡的參數）
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "password",
}


def start_docker():
    """
    確保 pgvector 容器有在跑。
    如果容器不存在就建立，如果已停止就重新啟動。
    """
    print("\n🐳 檢查 pgvector Docker 容器...")

    # 檢查容器是否存在
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=pgvector", "--format", "{{.Status}}"],
        capture_output=True, text=True
    )
    status = result.stdout.strip()

    if not status:
        # 容器不存在，建立並啟動
        print("  建立 pgvector 容器...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", "pgvector",
            "-e", "POSTGRES_PASSWORD=password",
            "-p", "5432:5432",
            "pgvector/pgvector:pg16"
        ], check=True)
        print("  等待 PostgreSQL 啟動...", end="", flush=True)
        time.sleep(5)
        print(" ✓")
    elif "Exited" in status:
        # 容器存在但已停止，重新啟動
        print("  重新啟動 pgvector 容器...")
        subprocess.run(["docker", "start", "pgvector"], check=True)
        time.sleep(3)
    else:
        print("  pgvector 容器已在運行 ✓")


def get_connection():
    """建立 PostgreSQL 連線，失敗時最多重試 5 次。"""
    for i in range(5):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except psycopg2.OperationalError:
            print(f"  等待資料庫就緒... ({i+1}/5)")
            time.sleep(3)
    raise RuntimeError("無法連接到 pgvector 資料庫")


def setup():
    """建立 pgvector 資料表並插入向量資料。"""
    start_docker()
    print("\n🟢 pgvector 設定中...")

    conn = get_connection()
    cur = conn.cursor()

    # 啟用 pgvector 擴充（只需要執行一次，之後資料庫就有 vector 型別）
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 建立資料表，vector(384) 代表儲存 384 維的向量
    cur.execute("DROP TABLE IF EXISTS documents;")
    cur.execute(f"""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            text TEXT,
            source TEXT,
            category TEXT,
            embedding vector({EMBEDDING_DIM})
        );
    """)

    # 小資料集（< 1000 筆）不建 IVFFlat 索引，直接用精確搜尋（sequential scan）
    # IVFFlat 適合大資料集（萬筆以上），資料太少時反而會漏掉結果

    conn.commit()

    docs = load_documents()
    texts = [d["text"] for d in docs]
    vectors = get_embeddings(texts)

    # 批次 INSERT，execute_values 比逐筆 INSERT 快很多
    t_start = time.time()
    rows = [
        (d["id"], d["text"], d["source"], d["category"], v)
        for d, v in zip(docs, vectors)
    ]
    execute_values(
        cur,
        "INSERT INTO documents (id, text, source, category, embedding) VALUES %s",
        rows,
        template="(%s, %s, %s, %s, %s::vector)",
    )
    conn.commit()
    elapsed = time.time() - t_start

    print(f"✅ pgvector 寫入 {len(docs)} 筆，耗時 {elapsed:.2f}s")
    cur.close()
    return conn, elapsed


def query(conn, question: str, n: int = 3) -> dict:
    """
    用餘弦距離（<=>）找最相似的 n 筆文件。
    PostgreSQL 的向量搜尋語法：embedding <=> query_vector
    <=> 是 pgvector 定義的運算子，代表「餘弦距離」（越小越相似）。
    """
    q_vector = get_embeddings([question])[0]
    q_vector_str = "[" + ",".join(str(x) for x in q_vector) + "]"

    cur = conn.cursor()
    t_start = time.time()
    cur.execute(f"""
        SELECT text, source, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (q_vector_str, q_vector_str, n))
    rows = cur.fetchall()
    elapsed = time.time() - t_start
    cur.close()

    return {
        "db": "pgvector",
        "question": question,
        "results": [row[0] for row in rows],
        "sources": [row[1] for row in rows],
        "scores": [round(row[2], 4) for row in rows],
        "query_time": elapsed,
    }


if __name__ == "__main__":
    conn, ingest_time = setup()

    print("\n🔍 查詢測試：")
    for q in TEST_QUERIES:
        r = query(conn, q)
        print(f"\n問：{r['question']}")
        print(f"來源：{r['sources']}")
        print(f"相似度分數：{r['scores']}")
        print(f"查詢耗時：{r['query_time']*1000:.1f}ms")
        print(f"第一段結果：{r['results'][0][:100]}...")

    conn.close()
