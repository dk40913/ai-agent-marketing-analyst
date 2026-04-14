"""
agents/workers/analysis_agent.py

Analysis Agent：負責原因分析。
接收 Data Agent 的數字摘要，再查 ChromaDB 知識庫，
把「數字異常」跟「背景知識」結合，給出有根據的分析。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from rag.query import rag_query

from config.llm import llm, safe_invoke, safe_invoke_full


# ── RAG 查詢包成 Tool ─────────────────────────────────────────────────────────
# Phase A 的 query_rag Node 直接呼叫 rag_query()
# Phase B 把它包成 @tool，讓 LLM 自己決定要不要查、查什麼問題

@tool
def search_knowledge_base(query: str) -> str:
    """
    從知識庫搜尋相關資訊，包含品牌介紹、市場分析報告、歷史異常事件紀錄。
    適合用來分析銷售異常原因、了解品牌背景、查詢過去發生過的問題。

    query: 搜尋關鍵字或問題描述
    """
    result = rag_query(query, n_results=3)
    return f"來源：{result['sources']}\n\n{result['answer']}"


TOOLS = [search_knowledge_base]
llm_with_tools = llm.bind_tools(TOOLS)


# ── Analysis Agent Node ───────────────────────────────────────────────────────

def analysis_agent(state: dict) -> dict:
    """
    結合數字數據與知識庫，分析異常原因。

    流程：
      1. 把數字摘要（data_result）+ 原始問題一起送給 LLM
      2. LLM 判斷是否需要查知識庫（search_knowledge_base）
      3. 整合數字 + 知識，生成原因分析
    """
    question = state.get("question", "")
    data_result = state.get("data_result", "")
    print(f"  [analysis_agent] 開始分析...")

    # 把數字摘要帶進 prompt，讓 LLM 知道「已經有哪些數字了」
    # 這樣 LLM 才能判斷「數字哪裡異常」，再決定要不要查知識庫補充原因
    prompt = f"""你是一位行銷數據分析師，請根據以下數據摘要分析異常原因。

【原始問題】
{question}

【數據摘要】
{data_result if data_result else '（尚無數據，請先查詢）'}

如果需要了解品牌背景、市場趨勢或歷史異常事件，請使用 search_knowledge_base 工具查詢。"""

    messages = [HumanMessage(content=prompt)]
    response = safe_invoke_full(llm_with_tools, messages)

    # 執行 LLM 決定呼叫的 Tool
    knowledge_results = []
    if response is None:
        return {**state, "analysis_result": data_result}
    if response.tool_calls:
        tool_map = {tool_fn.name: tool_fn for tool_fn in TOOLS}
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  [analysis_agent] 查詢知識庫：{tool_args.get('query', '')[:40]}...")
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)  # noqa: tool_fn
                knowledge_results.append(result)

    # 整合數字 + 知識庫，生成最終分析
    if knowledge_results:
        combined_knowledge = "\n\n".join(knowledge_results)
        final_prompt = f"""根據數據摘要和知識庫資訊，給出完整的原因分析。

【數據摘要】
{data_result}

【知識庫資訊】
{combined_knowledge}

【原始問題】
{question}

請給出有根據、具體的分析，說明數據異常的可能原因。"""

        analysis_result = safe_invoke(llm, [HumanMessage(content=final_prompt)], fallback=data_result)
    else:
        # LLM 判斷不需要查知識庫，直接用數字做分析
        # response.content 在某些 LangChain 版本下可能是 list（content blocks），
        # 需要正規化成純字串
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        analysis_result = content or data_result

    print(f"  [analysis_agent] 完成")
    return {**state, "analysis_result": analysis_result}
