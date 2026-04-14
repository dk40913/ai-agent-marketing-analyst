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

def resolve_question(question: str, history_text: str) -> str:
    """
    兩步驟問題解析：
      Step 1：讓 LLM 判斷問題是否有指代不明（代詞、「這家」「前面那個」等）
      Step 2：有指代不明時，根據對話歷史改寫成帶明確品牌名的問題

    比起用固定代詞清單判斷，LLM 能處理清單覆蓋不到的說法。
    第一輪對話（沒有歷史）直接跳過，不做任何改寫。
    """
    # 第一輪沒有歷史，無從解析，直接回傳
    if "（這是第一輪對話）" in history_text:
        return question

    # Step 1：讓 LLM 判斷問題是否有指代不明
    check_prompt = f"""這個問題是否有指代不明的地方？
指代不明的意思是：問題裡有「他」「它」「那個品牌」「這家」「前面那個」等需要靠上下文才能知道指的是誰的詞。
只回答 yes 或 no，不要解釋。

問題：{question}"""

    has_ambiguity = safe_invoke(llm, [HumanMessage(content=check_prompt)], fallback="no").strip().lower()

    if "yes" not in has_ambiguity:
        return question  # 問題清楚，不需要改寫

    # Step 2：根據對話歷史改寫問題，把代詞換成明確名稱
    rewrite_prompt = f"""根據以下對話歷史，把問題裡的代詞替換成明確的品牌名稱或主題。
只輸出改寫後的問題，不要解釋，不要加任何標點以外的文字。

【對話歷史】
{history_text}

【原始問題】
{question}

【改寫後的問題】"""

    resolved = safe_invoke(llm, [HumanMessage(content=rewrite_prompt)], fallback=question).strip()

    # 防禦：改寫後仍含代詞，代表 LLM 沒改成功，退回原問題
    PRONOUNS = ["他", "它", "她", "這個品牌", "那個品牌"]
    if any(p in resolved for p in PRONOUNS):
        return question

    print(f"  [supervisor] 問題改寫：「{question}」→「{resolved}」")
    return resolved


def supervisor(state: SupervisorState) -> SupervisorState:
    question = state["question"]
    data_result = state.get("data_result", "")
    analysis_result = state.get("analysis_result", "")
    report = state.get("report", "")

    # ── Problem 2 修正：從 messages 建立對話歷史 ──────────────────────────────
    # messages 裡存著之前幾輪的 HumanMessage / AIMessage
    # 把最近 4 則（2 輪）整理成文字，讓 Supervisor 理解「那」「它」等代詞指的是什麼
    history_lines = []
    for msg in state.get("messages", [])[-4:]:
        if hasattr(msg, "content"):
            role = "使用者" if msg.__class__.__name__ == "HumanMessage" else "AI"
            # 只取前 150 字，避免 prompt 過長
            history_lines.append(f"{role}：{msg.content[:150]}")
    history_text = "\n".join(history_lines) if history_lines else "（這是第一輪對話）"

    # 把目前已完成的工作列出來，讓 LLM 知道進度
    completed = []
    if data_result:
        completed.append("✅ 數據查詢（data_agent）")
    if analysis_result:
        completed.append("✅ 原因分析（analysis_agent）")
    if report:
        completed.append("✅ 報告產出（report_agent）")

    completed_text = "\n".join(completed) if completed else "（尚未開始）"

    # 第一次呼叫（data_result 還是空的）才做問題解析，避免每輪都重複呼叫 LLM
    if not data_result:
        question = resolve_question(question, history_text)
        # 把解析後的問題寫回 state，Worker 才能拿到
        state = {**state, "question": question}

    prompt = f"""你是一個 Multi-Agent 系統的 Supervisor。
根據使用者問題、對話歷史和目前完成的工作，決定下一步要呼叫哪個 Worker。

【對話歷史（最近幾輪）】
{history_text}

【本輪使用者問題】
{question}

【已完成的工作】
{completed_text}

【可用的 Worker】
- "data"：查詢銷售數據和廣告數據（需要數字時用）
- "analysis"：分析異常原因，結合知識庫（需要解釋原因時用）
- "report"：產出結構化報告（最後整理時用，需要先有數據和分析）
- "end"：所有工作完成，結束流程

規則：
1. 預設流程是 data → analysis → report → end
2. 只有問題是「純數字查詢」（例如：只問某品牌的銷售額數字、某個 KPI 的數值），才可以 data → end 跳過 analysis
3. 只要問題包含「原因」「為什麼」「分析」「建議」「比較」「改善」「下滑」「異常」等字眼，必須走完整流程
4. report 必須在 data 和 analysis 都完成後才能執行
5. 如果問題有代詞（「那」「它」「這個品牌」），請參考對話歷史找到實際指的品牌或主題
6. 所有需要的 Worker 都完成後，輸出 "end"

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
