"""
agents/supervisor.py

Supervisor + Multi-Agent 主圖。

架構：
  Supervisor 是中心 Node（Hub），Worker 是輻條（Spoke）。
  Supervisor 決定叫誰 → Worker 執行 → 回到 Supervisor → 決定下一步或結束。

State 設計：
  Phase B 比 Phase A 多了 worker_results（收集各 Worker 的輸出）
  和 next_worker（Supervisor 決定的下一個 Worker）。
"""

import os
import sys
from typing import TypedDict, Literal

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from config.llm import llm, safe_invoke
from agents.workers.data_agent import data_agent
from agents.workers.analysis_agent import analysis_agent
from agents.workers.report_agent import report_agent
from agents.workers.reviewer_agent import reviewer_agent


# ── State 定義 ────────────────────────────────────────────────────────────────
# Phase B 比 Phase A 多了三個欄位：
#   data_result     → Data Agent 的查詢結果
#   analysis_result → Analysis Agent 的分析結果
#   report          → Report Agent 的最終報告
#   next_worker     → Supervisor 決定的下一個 Worker，用來做 Conditional Edge 路由

class SupervisorState(TypedDict):
    question: str           # 使用者問題
    messages: list          # 對話歷史
    data_result: str        # Data Agent 輸出
    analysis_result: str    # Analysis Agent 輸出
    report: str             # Report Agent 輸出
    next_worker: str        # 下一個要執行的 Worker（"data"/"analysis"/"report"/"end"）
    final_answer: str       # 整合後的最終回答


# ── Supervisor Node ───────────────────────────────────────────────────────────
# Supervisor 的核心邏輯：看目前 State 有哪些結果，決定下一步叫誰

def supervisor(state: SupervisorState) -> SupervisorState:
    question = state["question"]
    data_result = state.get("data_result", "")
    analysis_result = state.get("analysis_result", "")
    report = state.get("report", "")

    # 把目前已完成的工作列出來，讓 LLM 知道進度
    completed = []
    if data_result:
        completed.append("✅ 數據查詢（data_agent）")
    if analysis_result:
        completed.append("✅ 原因分析（analysis_agent）")
    if report:
        completed.append("✅ 報告產出（report_agent）")

    completed_text = "\n".join(completed) if completed else "（尚未開始）"

    prompt = f"""你是一個 Multi-Agent 系統的 Supervisor。
根據使用者問題和目前完成的工作，決定下一步要呼叫哪個 Worker。

【使用者問題】
{question}

【已完成的工作】
{completed_text}

【可用的 Worker】
- "data"：查詢銷售數據和廣告數據（需要數字時用）
- "analysis"：分析異常原因，結合知識庫（需要解釋原因時用）
- "report"：產出結構化報告（最後整理時用，需要先有數據和分析）
- "end"：所有工作完成，結束流程

規則：
1. 通常順序是 data → analysis → report → end
2. 簡單的數字查詢可以只做 data → end
3. report 必須在 data 和 analysis 都完成後才能執行
4. 所有需要的 Worker 都完成後，輸出 "end"

只輸出一個單字：data、analysis、report 或 end。不要解釋。"""

    next_worker = safe_invoke(llm, [HumanMessage(content=prompt)], fallback="end").strip().lower()

    # 防禦一：如果 LLM 輸出不在預期範圍，預設結束
    if next_worker not in ["data", "analysis", "report", "end"]:
        next_worker = "end"

    # 防禦二：report 必須在 data 和 analysis 都完成後才能執行
    # LLM 有可能忽略 prompt 裡的規則，這裡用程式碼強制執行
    if next_worker == "report" and (not data_result or not analysis_result):
        print(f"  [supervisor] 攔截：report 的前置條件未達成，改為 data")
        next_worker = "data" if not data_result else "analysis"

    print(f"  [supervisor] 下一步：{next_worker}")
    return {**state, "next_worker": next_worker}


# ── Conditional Edge：Supervisor 決定路由 ────────────────────────────────────

def route_to_worker(state: SupervisorState) -> Literal["data_agent", "analysis_agent", "report_agent", "generate_final"]:
    mapping = {
        "data": "data_agent",
        "analysis": "analysis_agent",
        "report": "report_agent",
        "end": "generate_final",
    }
    return mapping.get(state["next_worker"], "generate_final")


# ── 最終整合 Node ─────────────────────────────────────────────────────────────
# 把 report（或 data_result）包裝成最終回答，存進 messages

def generate_final(state: SupervisorState) -> SupervisorState:
    # 有報告就用報告，沒有就用數據摘要（簡單問題不一定需要完整報告）
    final_answer = state.get("report") or state.get("analysis_result") or state.get("data_result", "查無結果")

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=final_answer),
    ]

    print(f"  [generate_final] 完成")
    return {**state, "final_answer": final_answer, "messages": new_messages}


# ── 建立並編譯 Multi-Agent Graph ──────────────────────────────────────────────

def build_supervisor():
    graph = StateGraph(SupervisorState)

    # 加入所有 Node
    graph.add_node("supervisor", supervisor)
    graph.add_node("data_agent", data_agent)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("generate_final", generate_final)

    # 入口點：從 Supervisor 開始
    graph.set_entry_point("supervisor")

    # Supervisor → Conditional Edge → 各個 Worker 或結束
    graph.add_conditional_edges("supervisor", route_to_worker)

    # 每個 Worker 執行完都回到 Supervisor
    # 這是 Hub and Spoke 的關鍵：Worker 不直接連到下一個 Worker
    graph.add_edge("data_agent", "supervisor")
    graph.add_edge("analysis_agent", "reviewer_agent")   # 分析完先過審核
    graph.add_edge("reviewer_agent", "supervisor")        # 審核完再回 supervisor
    graph.add_edge("report_agent", "supervisor")

    # generate_final 是終點
    graph.add_edge("generate_final", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


multi_agent = build_supervisor()


# ── 互動式測試 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "supervisor-demo-1"}}

    test_questions = [
        "幫我查鮮橙生活的總銷售額",
        "山嵐咖啡銷售下滑，可能原因是什麼？",
        "幫我分析本季表現最差的品牌，並給出改善建議",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"問：{q}")
        result = multi_agent.invoke(
            {
                "question": q,
                "messages": [],
                "data_result": "",
                "analysis_result": "",
                "report": "",
                "next_worker": "",
                "final_answer": "",
            },
            config=config,
        )
        print(f"\n答：{result['final_answer']}")
