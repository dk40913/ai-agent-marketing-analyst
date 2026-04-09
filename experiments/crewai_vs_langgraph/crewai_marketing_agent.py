"""
experiments/crewai_vs_langgraph/crewai_marketing_agent.py

用 CrewAI 重新實作和 Phase B 相同的 Multi-Agent 邏輯：
  Data Agent → Analysis Agent → Reviewer Agent → Report Agent

目的：對比 LangGraph 和 CrewAI 在同一套邏輯下的寫法差異。
這個檔案不進主系統，只作為比較實驗。
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import Field

from tools.query_sales import query_sales as _query_sales
from tools.query_ad_spend import query_ad_spend as _query_ad_spend
from rag.query import rag_query

# ── LLM 設定 ─────────────────────────────────────────────────────────────────
# CrewAI 1.x 用 LiteLLM，llm 參數要傳模型名稱字串，格式是 "provider/model"
# 不接受 LangChain 的 ChatGoogleGenerativeAI 等物件
_provider = os.environ.get("LLM_PROVIDER", "groq").lower()
if _provider == "gemini":
    CREWAI_LLM = "gemini/gemma-4-26b-a4b-it"
else:
    CREWAI_LLM = "groq/llama-3.3-70b-versatile"


# ── CrewAI Tool 包裝 ──────────────────────────────────────────────────────────
# CrewAI 1.x 的 Tool 系統和 LangChain 不同，要繼承 BaseTool 重新包裝。
# 底層邏輯（查 CSV、查 ChromaDB）直接呼叫 Phase B 的函式，不重複寫。

class QuerySalesTool(BaseTool):
    name: str = "query_sales"
    description: str = (
        "查詢品牌銷售數據摘要，回傳總銷售額、總數量、平均退貨率等指標。"
        "brand 可為空（查全部），date_from/date_to 格式 YYYY-MM-DD 可為空。"
    )

    def _run(self, brand: str = "", date_from: str = "", date_to: str = "") -> str:
        return _query_sales.invoke({"brand": brand, "date_from": date_from, "date_to": date_to})


class QueryAdSpendTool(BaseTool):
    name: str = "query_ad_spend"
    description: str = (
        "查詢品牌廣告投放數據，回傳總花費、ROAS、點擊率、轉換率等指標。"
        "brand 可為空（查全部），date_from/date_to 格式 YYYY-MM-DD 可為空。"
    )

    def _run(self, brand: str = "", date_from: str = "", date_to: str = "") -> str:
        return _query_ad_spend.invoke({"brand": brand, "date_from": date_from, "date_to": date_to})


class SearchKnowledgeBaseTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "從知識庫搜尋相關資訊，包含品牌介紹、市場分析報告、歷史異常事件紀錄。"
        "適合分析銷售異常原因、了解品牌背景、查詢過去發生過的問題。"
    )

    def _run(self, query: str) -> str:
        result = rag_query(query, n_results=3)
        return f"來源：{result['sources']}\n\n{result['answer']}"


# ── 定義 Agents（角色）────────────────────────────────────────────────────────
# LangGraph：Node = 一個處理函式
# CrewAI：Agent = 有名字、角色說明、工具的「人物」

data_analyst = Agent(
    role="行銷數據分析師",
    goal="從銷售數據和廣告數據中找出關鍵指標和異常",
    backstory="""你是一位有五年經驗的行銷數據分析師，熟悉 ROAS、CTR、CVR 等指標。
你擅長從數字中發現趨勢和異常，並用清楚的文字呈現數據摘要。""",
    tools=[QuerySalesTool(), QueryAdSpendTool()],
    llm=CREWAI_LLM,
    verbose=True,
)

analysis_expert = Agent(
    role="行銷策略顧問",
    goal="結合數字數據和市場知識，分析銷售異常的根本原因",
    backstory="""你是一位資深行銷策略顧問，除了解讀數字，你更善於結合品牌背景、
市場趨勢和歷史事件，給出有深度的原因分析。""",
    tools=[SearchKnowledgeBaseTool()],
    llm=CREWAI_LLM,
    verbose=True,
)

reviewer = Agent(
    role="數據品管審核員",
    goal="審核分析報告是否符合業務常理，過濾不合邏輯的結論",
    backstory="""你是一位嚴格的數據品管審核員，熟知行銷數據的業務常理。
你的工作是找出分析報告中違反常理的結論，例如負數的 ROAS、超過 100% 的退貨率等。
只有在明確違反常理時才提出修正，不過度挑剔。""",
    tools=[],
    llm=CREWAI_LLM,
    verbose=True,
)

report_writer = Agent(
    role="行銷報告撰寫員",
    goal="把分析結果整理成結構清楚、易於閱讀的行銷分析報告",
    backstory="""你是一位專業的行銷報告撰寫員，擅長把複雜的數據分析轉化為
清楚的執行摘要、具體發現和可行的改善建議。""",
    tools=[],
    llm=CREWAI_LLM,
    verbose=True,
)


# ── 定義 Tasks（任務）────────────────────────────────────────────────────────
# LangGraph：Node 函式自己決定做什麼，Supervisor 決定執行順序
# CrewAI：Task 預先定義「做什麼、期望輸出什麼」，Crew 按依賴關係執行

def build_crew(question: str) -> Crew:
    """
    根據使用者問題動態建立 Crew。
    Task 的 description 會帶入問題，讓各 Agent 知道要分析什麼。
    """

    task_data = Task(
        description=f"""查詢以下問題相關的銷售和廣告數據：
{question}

使用 query_sales 和 query_ad_spend 工具查詢相關品牌數據。
如果問題沒有指定品牌，查詢所有品牌。""",
        expected_output="包含總銷售額、廣告 ROAS、各通路表現的數據摘要",
        agent=data_analyst,
    )

    task_analysis = Task(
        description=f"""根據數據摘要，分析以下問題的原因：
{question}

請使用 search_knowledge_base 查詢品牌背景和歷史事件，
結合數字和知識庫，給出有根據的原因分析。""",
        expected_output="詳細的原因分析，說明數據異常的可能原因，並引用知識庫來源",
        agent=analysis_expert,
        context=[task_data],  # 依賴 task_data 的結果
    )

    task_review = Task(
        description="""審核上一步的分析報告是否符合業務常理：
1. ROAS 不可能為負數（正常範圍 0.5～10）
2. 單日銷售額成長不可能超過 500%
3. 退貨率不可能超過 100%
4. 數字引用必須和數據摘要一致

如果發現問題，指出具體錯誤並提供修正。
如果沒有問題，確認分析合理並輸出原始分析。""",
        expected_output="審核結果：APPROVED（附說明）或 REVISED（附修正內容）",
        agent=reviewer,
        context=[task_data, task_analysis],  # 需要看數據和分析
    )

    task_report = Task(
        description=f"""根據數據摘要、原因分析和審核結果，產出完整的行銷分析報告。

原始問題：{question}

報告格式：
## 執行摘要
## 數據發現
## 原因分析
## 改善建議""",
        expected_output="結構完整的行銷分析報告，包含執行摘要、數據發現、原因分析、改善建議",
        agent=report_writer,
        context=[task_data, task_analysis, task_review],  # 整合全部結果
    )

    return Crew(
        agents=[data_analyst, analysis_expert, reviewer, report_writer],
        tasks=[task_data, task_analysis, task_review, task_report],
        process=Process.sequential,  # 按順序執行，不需要 Supervisor
        verbose=True,
    )


# ── 執行入口 ──────────────────────────────────────────────────────────────────

def run_crewai(question: str) -> str:
    """執行 CrewAI Multi-Agent，回傳最終報告。"""
    crew = build_crew(question)
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    questions = [
        "幫我查鮮橙生活的總銷售額",
        "山嵐咖啡銷售下滑，可能原因是什麼？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"問：{q}")
        answer = run_crewai(q)
        print(f"\n答：{answer}")
