"""
agents/workers/data_agent.py

Data Agent：負責查詢結構化數據（銷售 + 廣告）。

Phase A 是 Node 直接讀 CSV，Phase B 改為：
  LLM 拿到 Tool 清單 → 自己判斷要呼叫哪個 Tool → Tool 查詢並回傳結果
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config.llm import llm, safe_invoke, safe_invoke_full
from tools.query_sales import query_sales
from tools.query_ad_spend import query_ad_spend

# bind_tools：把工具清單綁給 LLM
# LLM 收到問題後，如果判斷需要查數據，會在回應裡附上 tool_calls
# tool_calls 是 LLM 說「我要呼叫這個工具、傳這些參數」的指令
TOOLS = [query_sales, query_ad_spend]
llm_with_tools = llm.bind_tools(TOOLS)


# ── Data Agent Node ───────────────────────────────────────────────────────────

def data_agent(state: dict) -> dict:
    """
    接收 Supervisor 分配的任務，用 Tool 查詢數據，回傳結果。

    流程：
      1. LLM 判斷問題需要哪些 Tool、傳什麼參數
      2. 執行 Tool，拿到查詢結果
      3. LLM 整合結果，生成文字摘要
    """
    question = state.get("current_task") or state.get("question", "")
    print(f"  [data_agent] 收到任務：{question[:50]}...")

    # Step 1：LLM 判斷要呼叫哪些 Tool
    messages = [HumanMessage(content=question)]
    response = safe_invoke_full(llm_with_tools, messages)

    # Step 2：執行 LLM 決定呼叫的 Tool
    # response.tool_calls 是一個 list，每個元素是 {"name": "function名稱", "args": {...}}
    tool_results = []
    if response is None:
        return {**state, "data_result": f"（data_agent LLM 呼叫失敗，問題：{question[:30]}）"}
    if response.tool_calls:
        # 建立 Tool 名稱到函式的對應表，方便查找
        tool_map = {tool.name: tool for tool in TOOLS}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  [data_agent] 呼叫 Tool：{tool_name}，參數：{tool_args}")

            if tool_name in tool_map:
                # 執行 Tool，傳入 LLM 決定的參數
                result = tool_map[tool_name].invoke(tool_args)
                tool_results.append(f"【{tool_name} 結果】\n{result}")

    # Step 3：如果有 Tool 結果，讓 LLM 整合成摘要
    if tool_results:
        combined = "\n\n".join(tool_results)
        summary_prompt = f"""根據以下查詢結果，整理成清楚的數據摘要，供後續分析使用。

{combined}

原始問題：{question}"""
        # safe_invoke：遇到 rate limit 會自動等待重試，失敗才用 combined 當 fallback
        data_result = safe_invoke(llm, [HumanMessage(content=summary_prompt)], fallback=combined)
    else:
        # LLM 沒有呼叫 Tool 時，確保 data_result 不是空字串
        data_result = response.content or f"（data_agent 已執行，問題：{question[:30]}）"

    print(f"  [data_agent] 完成")
    return {**state, "data_result": data_result}
