"""
Phase A：第一個 LangGraph Single Agent

StateGraph 三元素：
  - State：記錄目前對話狀態的「記憶板」，每個 Node 都能讀取與更新
  - Node：每個處理步驟的函式
  - Edge：決定下一步走哪裡（Conditional Edge 讓 Agent 自己選路徑）

流程：
  使用者問題 → [classify] → Conditional Edge → [query_data] 或 [query_rag] → [generate_answer]
"""

import os
import sys
import pandas as pd
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# 讓這個腳本能 import 專案其他模組（rag/query.py 等）
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
from rag.query import rag_query
from config.llm import llm

CSV_DIR = os.path.join(BASE_DIR, "data", "csv")

# ── State 定義 ────────────────────────────────────────────────────────────────
# TypedDict 讓每個欄位都有型別，方便 IDE 提示也讓程式更好讀
# 每個 Node 只能「新增或覆寫」State 的欄位，不能刪除

class AgentState(TypedDict):
    messages: list       # 多輪對話歷史，Checkpointer 用來跨輪記憶
    question: str        # 使用者當前問題
    question_type: str   # "data" 或 "knowledge"，由 classify node 填入
    detected_brand: str  # 從問題或對話歷史解析出的品牌名稱，空字串代表全部品牌
    query_result: str    # 查詢結果，由 query_data 或 query_rag 填入
    answer: str          # 最終回答，由 generate_answer 填入


# ── Node 1：classify ──────────────────────────────────────────────────────────
# 讓 LLM 判斷這個問題要查「結構化數據（CSV）」還是「知識庫（ChromaDB RAG）」
# 這就是 Conditional Edge 的依據

def classify_question(state: AgentState) -> AgentState:
    question = state["question"]

    # 把對話歷史轉成文字，讓 LLM 理解「它」、「這個品牌」等代詞
    history_text = ""
    for msg in state.get("messages", []):
        role = "使用者" if isinstance(msg, HumanMessage) else "助理"
        history_text += f"{role}：{msg.content}\n"

    prompt = f"""你是一個問題分析器。根據對話歷史和當前問題，輸出以下兩項資訊：

    第一行：資料來源類型，只能是 "data" 或 "knowledge"
    - "data"：問題涉及可量化的數字欄位，包括：銷售額、數量、退貨率、廣告花費、ROAS、CTR、轉換數、通路業績。即使問法是「是否正常」、「表現如何」，只要核心是數字指標就選 data
    - "knowledge"：問題涉及原因分析、策略建議、品牌背景介紹、市場趨勢解讀（不是在問具體數字）

    第二行：品牌名稱，只能是 "鮮橙生活"、"山嵐咖啡"、"城市動力" 其中一個，或 "none"（如果問題與特定品牌無關）

    【對話歷史】
    {history_text if history_text else "（無）"}

    【當前問題】
    {question}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    lines = response.content.strip().splitlines()

    question_type = lines[0].strip().lower() if lines else "knowledge"
    if question_type not in ["data", "knowledge"]:
        question_type = "knowledge"

    detected_brand = lines[1].strip() if len(lines) > 1 else "none"
    if detected_brand not in ["鮮橙生活", "山嵐咖啡", "城市動力"]:
        detected_brand = ""

    print(f"  [classify] → {question_type}，品牌：{detected_brand or '全部'}")
    return {**state, "question_type": question_type, "detected_brand": detected_brand}


# ── Node 2a：query_data ───────────────────────────────────────────────────────
# 直接讀 CSV，算出摘要數字給 LLM 參考
# 不走 FastAPI 是因為 Phase A 重點是學 LangGraph，不想為了查數據還要另外啟 server

def query_data(state: AgentState) -> AgentState:
    # 直接使用 classify 解析好的品牌，不再自己從問題文字偵測
    detected_brand = state.get("detected_brand", "")

    sales_df = pd.read_csv(os.path.join(CSV_DIR, "sales_data.csv"))
    ad_df = pd.read_csv(os.path.join(CSV_DIR, "ad_spend.csv"))

    s = sales_df[sales_df["brand_name"] == detected_brand] if detected_brand else sales_df
    a = ad_df[ad_df["brand_name"] == detected_brand] if detected_brand else ad_df

    result = f"""銷售摘要（{detected_brand or '全品牌'}）：
- 總銷售額：{int(s['sales_amount'].sum()):,}
- 總數量：{int(s['quantity'].sum()):,}
- 平均退貨率：{s['return_rate'].mean():.2%}
- 資料筆數：{len(s)} 筆

廣告摘要：
- 總花費：{a['spend'].sum():,.0f}
- 平均 ROAS：{a['roas'].mean():.2f}
- 總轉換數：{int(a['conversions'].sum()):,}"""

    print(f"  [query_data] 品牌：{detected_brand or '全部'}")
    return {**state, "query_result": result}


# ── Node 2b：query_rag ────────────────────────────────────────────────────────
# 呼叫 Phase 0 建好的 RAG，從 ChromaDB 找最相關的知識庫段落
# 這裡直接複用 rag/query.py 的函式，不重複造輪子

def query_rag(state: AgentState) -> AgentState:
    result = rag_query(state["question"], n_results=3)
    query_result = f"來源：{result['sources']}\n\n{result['answer']}"
    print(f"  [query_rag] 來源：{result['sources']}")
    return {**state, "query_result": query_result}


# ── Node 3：generate_answer ───────────────────────────────────────────────────
# 把查詢結果整合成自然語言回答，並把這輪問答記入 messages

def generate_answer(state: AgentState) -> AgentState:
    prompt = f"""你是一位專業的行銷數據分析師，服務三個品牌客戶：鮮橙生活、山嵐咖啡、城市動力。

根據以下查詢結果，用繁體中文回答使用者問題。回答要具體、有條理。
如果資料不足，直接說明不足之處，不要猜測。

【查詢結果】
{state['query_result']}

【使用者問題】
{state['question']}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    # messages 累積對話歷史，讓 Checkpointer 跨輪記憶
    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=answer),
    ]

    print(f"  [generate_answer] 完成")
    return {**state, "answer": answer, "messages": new_messages}


# ── Conditional Edge：classify 後根據 question_type 決定下一個 Node ────────────
# 這個函式的回傳值必須對應到 add_conditional_edges 裡定義的 key

def route_by_type(state: AgentState) -> Literal["query_data", "query_rag"]:
    return "query_data" if state["question_type"] == "data" else "query_rag"


# ── 建立並編譯 StateGraph ─────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    # 加入所有 Node
    graph.add_node("classify", classify_question)
    graph.add_node("query_data", query_data)
    graph.add_node("query_rag", query_rag)
    graph.add_node("generate_answer", generate_answer)

    # 入口點
    graph.set_entry_point("classify")

    # Conditional Edge：classify → route_by_type() → query_data 或 query_rag
    graph.add_conditional_edges("classify", route_by_type)

    # 固定 Edge
    graph.add_edge("query_data", "generate_answer")
    graph.add_edge("query_rag", "generate_answer")
    graph.add_edge("generate_answer", END)

    # MemorySaver：把 State 存在記憶體
    # 同一個 thread_id 的多輪對話會共享同一份 State（Checkpointer 的作用）
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


agent = build_agent()


# ── 互動式測試 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # thread_id 是對話 session 的識別碼
    # 同一個 thread_id → Agent 記得之前說過什麼
    # 不同 thread_id → 全新對話，沒有記憶
    config = {"configurable": {"thread_id": "demo-session-1"}}

    test_questions = [
        "幫我查鮮橙生活的總銷售額",
        "山嵐咖啡銷售下滑，可能原因是什麼？",
        "城市動力的廣告 ROAS 表現如何？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"問：{q}")
        result = agent.invoke(
            {
                "question": q,
                "messages": [],
                "question_type": "",
                "query_result": "",
                "answer": "",
            },
            config=config,
        )
        print(f"答：{result['answer']}")
