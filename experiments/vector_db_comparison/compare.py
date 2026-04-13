"""
Phase D - 向量 DB 比較主程式

同樣的資料、同樣的查詢、三個 DB，並排比較：
  - 寫入速度（ingest time）
  - 查詢速度（query time）
  - 查詢結果是否一致（來源文件是否相同）

執行方式：
  cd experiments/vector_db_comparison
  python compare.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from shared import TEST_QUERIES
import chroma_demo
import pinecone_demo
import pgvector_demo


def run_all():
    print("=" * 60)
    print("Phase D：向量 DB 比較實驗")
    print("=" * 60)

    # ── 1. 初始化三個 DB ──────────────────────────────────────
    chroma_col, chroma_ingest = chroma_demo.setup()
    pinecone_idx, pinecone_ingest = pinecone_demo.setup()
    pg_conn, pg_ingest = pgvector_demo.setup()

    # ── 2. 寫入速度比較 ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 寫入速度比較")
    print("=" * 60)
    print(f"  ChromaDB : {chroma_ingest:.2f}s")
    print(f"  Pinecone : {pinecone_ingest:.2f}s  (含網路上傳)")
    print(f"  pgvector : {pg_ingest:.2f}s")

    # ── 3. 查詢結果比較 ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 查詢結果比較")
    print("=" * 60)

    query_times = {"ChromaDB": [], "Pinecone": [], "pgvector": []}

    for q in TEST_QUERIES:
        print(f"\n問：{q}")
        print("-" * 50)

        r_chroma = chroma_demo.query(chroma_col, q)
        r_pinecone = pinecone_demo.query(pinecone_idx, q)
        r_pg = pgvector_demo.query(pg_conn, q)

        query_times["ChromaDB"].append(r_chroma["query_time"])
        query_times["Pinecone"].append(r_pinecone["query_time"])
        query_times["pgvector"].append(r_pg["query_time"])

        # 顯示每個 DB 找到的來源文件
        print(f"  ChromaDB  來源：{r_chroma['sources']}")
        print(f"  Pinecone  來源：{r_pinecone['sources']}  分數：{r_pinecone['scores']}")
        print(f"  pgvector  來源：{r_pg['sources']}  分數：{r_pg['scores']}")

        # 檢查三個 DB 的結果是否一致（第一個來源相同即視為一致）
        tops = [r['sources'][0] if r['sources'] else '(無結果)' for r in [r_chroma, r_pinecone, r_pg]]
        if len(set(tops)) == 1:
            print("  ✅ 三個 DB 結果一致")
        else:
            print(f"  ⚠️  結果不同：{tops}")

    # ── 4. 查詢速度彙整 ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("⚡ 平均查詢速度")
    print("=" * 60)
    for db, times in query_times.items():
        avg_ms = sum(times) / len(times) * 1000
        print(f"  {db:<12}: {avg_ms:.1f}ms")

    # ── 5. 總結 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📝 選型建議")
    print("=" * 60)
    print("""
  ChromaDB  → 適合：快速原型、本機開發、資料量 < 10 萬
              不適合：生產環境大規模查詢

  Pinecone  → 適合：生產環境、免維護、需要高可用
              不適合：離線環境、對資料外流有顧慮

  pgvector  → 適合：已有 PostgreSQL、需混合 SQL + 向量查詢
              不適合：純向量搜尋、不熟 SQL 的團隊
    """)

    pg_conn.close()
    print("🎉 比較完成！")


if __name__ == "__main__":
    run_all()
