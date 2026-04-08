# 協作規則

## 教學風格
- 每寫一段新程式碼之前，先用 1–3 句話解釋「這段在做什麼、為什麼這樣做」
- 遇到新技術或術語（FastAPI、LangGraph、RAG、ChromaDB 等），在當下解釋清楚，不要假設我已經懂
- 說明要具體，避免抽象（例如「FastAPI 會幫你自動產生 /docs 頁面，你現在看到的 Swagger UI 就是這個」）

## 階段結束時存 Obsidian 筆記
- 每個實作階段（data、api、rag、agents、tools、experiments）完成後，自動將該階段的學習重點、程式碼說明、重要概念整理成筆記
- 筆記存放位置：`/Users/herb/Documents/Obsidian/AI Agent實作筆記/`
- 筆記格式：Obsidian Flavored Markdown（含 frontmatter、callout、wikilink）
- 筆記命名規則：`階段名稱 - 主題.md`（例如 `01 - 假資料生成.md`、`02 - FastAPI.md`）

## 筆記拆分原則
- **專案筆記**（`AI Agent實作筆記/`）：專注於實作細節、程式碼說明、這個專案裡的具體用法。遇到 AI 概念時簡短提及即可，加上 wikilink 指向 AI 筆記。
- **AI 筆記**（`/Users/herb/Documents/Obsidian/AI筆記/` 或 vault 根目錄）：每個值得獨立解釋的 AI 概念都應有一篇完整的獨立筆記，涵蓋定義、原理、比較、延伸。
- 寫專案筆記時，主動檢查有無 AI 概念尚未有獨立筆記，若沒有則一併建立。
- 兩層筆記透過 wikilink 互相連結，專案筆記是「怎麼用」，AI 筆記是「是什麼」。

---

# 專案背景

**目標職位**：AI Agent Engineer（invos 麻布數據科技）
**專案名稱**：AI Agent 行銷數據分析助手
**用途**：面試練習專案，模擬 invos 真實業務場景

模擬場景：品牌客戶輸入業務問題（如「上週銷售異常原因是什麼？」），Multi-Agent 系統自動拆解任務、查詢數據、分析、產出報告。

---

# Tech Stack

- **LLM**：Gemini API（免費額度，Key 已有）
- **Agent 框架**：LangGraph（主要）、CrewAI（比較用）
- **向量 DB**：ChromaDB（開發）→ Pinecone / pgvector（比較實驗）
- **API**：FastAPI + Uvicorn
- **資料**：Pandas（CSV）

---

# 測試資料（全部用 Python 生成，不需要真實資料）

**三個假品牌**：鮮橙生活（日用品）、山嵐咖啡（餐飲）、城市動力（運動）

**結構化資料（CSV）**
- `data/csv/sales_data.csv`：date, brand_name, category, channel, sales_amount, quantity, return_rate（約 810 筆）
- `data/csv/ad_spend.csv`：date, brand_name, platform, spend, impressions, clicks, conversions, roas（約 810 筆）

**非結構化文本（存入 ChromaDB）**
- `data/knowledge_base/brand_profiles/`：3 份品牌介紹（各約 500 字）
- `data/knowledge_base/market_reports/`：3 份季度分析報告（各約 1000 字）
- `data/knowledge_base/anomaly_logs/`：5 份歷史異常事件紀錄

**測試問句**
- 簡單：「幫我查鮮橙生活上個月的總銷售額」
- 中等：「山嵐咖啡銷售下滑，可能原因是什麼？」
- 複雜：「幫我分析本季表現最差的 3 個品牌，並給出改善建議」

---

# 系統架構

```
使用者問題
    ↓
[Supervisor Agent]  ← 理解問題、拆解任務、分配工作
    ↓           ↓              ↓
[Data Agent]  [Analysis Agent]  [Report Agent]
查詢銷售數據   分析異常原因       產出摘要報告
                  ↓
           [Reviewer Agent]  ← 審核計算邏輯是否符合業務常理
    ↓           ↓              ↓
         [Supervisor Agent]
         整合結果 → 最終回答
```

**Reviewer Agent 存在原因**：行銷數據有業務常理限制（ROAS 不可能為負數、銷售額不可能一天暴增 1000%），防止 LLM 幻覺導致誤導性結論。

---

# 目錄結構與用途

```
api/         → FastAPI 三個端點：/sales, /adspend, /brands
rag/         → ChromaDB ingest + 相似度查詢
agents/      → Supervisor + Workers（LangGraph 實作）
tools/       → Agent 可呼叫的工具函式
experiments/ → 不進主系統的比較實驗
  crewai_vs_langgraph/   → 用 CrewAI 重新實作同樣邏輯，比較差異
  vector_db_comparison/  → ChromaDB vs Pinecone vs pgvector
docs/        → 架構設計文件、向量 DB 選型筆記（面試素材）
data/        → 假資料與生成腳本
```

---

# 實作順序

1. `data/generate_data.py` — 生成所有假資料
2. `api/main.py` — FastAPI 三個端點
3. `rag/ingest.py` + `rag/query.py` — ChromaDB + RAG 跑通
4. `agents/` — 單一 Agent → Supervisor + Workers
5. `tools/` — 工具定義，供 Agent 呼叫 API
6. `experiments/` — CrewAI 比較、向量 DB 選型

---

# 最終交付物（面試用）

1. GitHub Repo：完整 code + README
2. `docs/architecture.md`：Multi-Agent 設計說明
3. `docs/vector_db_comparison.md`：向量 DB 選型筆記
4. Demo 影片：3 分鐘展示完整分析流程
