# AI Agent 行銷數據分析助手

> invos（麻布數據科技）AI Agent Engineer 面試練習專案

模擬真實業務場景：品牌客戶輸入業務問題，Multi-Agent 系統自動拆解任務、查詢數據、分析、產出報告。

---

## 專案架構

```
使用者問題
    ↓
[Supervisor Agent]  ← 負責理解問題、拆解任務、分配工作
    ↓           ↓            ↓
[Data Agent]  [Analysis Agent]  [Report Agent]
查詢銷售數據   分析異常原因      產出摘要報告
    ↓           ↓            ↓
         [Supervisor Agent]
         整合結果 → 最終回答
```

---

## 實作進度

| Phase | 內容 | 狀態 |
|-------|------|------|
| Phase 0 | RAG + FastAPI 基礎建設 | ✅ 完成 |
| Phase A | LangGraph 單一 Agent | ✅ 完成 |
| Phase B | Multi-Agent 協作架構 | ✅ 完成 |
| Phase C | 業務流程拆解思維 | ✅ 完成 |
| Phase D | 向量資料庫選型實作 | ✅ 完成 |

---

## 目錄說明

```
ai-agent-marketing-analyst/
├── data/                        # 測試資料（Python 生成）
│   ├── generate_data.py         # 假資料生成腳本
│   ├── csv/                     # sales_data.csv, ad_spend.csv
│   └── knowledge_base/          # 品牌介紹、市場報告、異常記錄
├── api/                         # FastAPI 服務（三個查詢端點）
│   └── main.py
├── rag/                         # ChromaDB + RAG 查詢
│   ├── ingest.py                # 文件切段存入向量 DB
│   └── query.py                 # 相似度查詢
├── agents/                      # Agent 定義
│   ├── supervisor.py
│   └── workers/
│       ├── data_agent.py
│       ├── analysis_agent.py
│       └── report_agent.py
├── tools/                       # Agent 工具（呼叫 API 或查詢 DB）
│   ├── query_sales.py
│   └── query_ad_spend.py
├── experiments/                 # 比較實驗（不進主系統）
│   ├── crewai_vs_langgraph/     # CrewAI 重新實作 + 比較
│   └── vector_db_comparison/    # Pinecone / pgvector / ChromaDB 比較
├── docs/                        # 設計文件 + 選型筆記
│   └── architecture.md
└── requirements.txt
```

---

## 測試問句

- **簡單**：「幫我查鮮橙生活上個月的總銷售額」
- **中等**：「山嵐咖啡銷售下滑，可能原因是什麼？」
- **複雜**：「幫我分析本季表現最差的 3 個品牌，並給出改善建議」

---

## Tech Stack

- **LLM**：Gemini API（免費額度）
- **Agent 框架**：LangGraph、CrewAI
- **向量 DB**：ChromaDB → Pinecone → pgvector
- **API**：FastAPI + Uvicorn
- **資料**：Pandas（CSV）+ ChromaDB（文本）
