"""
agents/workers/report_agent.py

Report Agent：負責產出結構化報告。
不需要查詢任何外部資料，純粹把前兩個 Agent 的結果
整理成清楚、有條理的報告格式。

所以這個 Agent 不需要 Tool，也不需要 bind_tools。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

from langchain_core.messages import HumanMessage

from config.llm import llm, safe_invoke


# ── Report Agent Node ─────────────────────────────────────────────────────────

def report_agent(state: dict) -> dict:
    """
    整合數據摘要與分析結果，產出結構化報告。

    報告格式：
      1. 摘要（一句話結論）
      2. 數據概況
      3. 異常原因分析
      4. 改善建議
    """
    question = state.get("question", "")
    data_result = state.get("data_result", "")
    analysis_result = state.get("analysis_result", "")
    print(f"  [report_agent] 產出報告...")

    prompt = f"""你是一位專業的行銷顧問，請根據以下資料產出一份結構化分析報告。

【原始問題】
{question}

【數據摘要】
{data_result if data_result else '（無數據）'}

【原因分析】
{analysis_result if analysis_result else '（無分析）'}

請按照以下格式產出報告：

## 摘要
（一句話說明核心結論）

## 數據概況
（關鍵指標與重點數字）

## 原因分析
（數據異常的可能原因，要有根據）

## 改善建議
（具體、可執行的行動建議，至少 2 點）

報告要具體、有數字支撐，避免模糊的建議。"""

    report = safe_invoke(llm, [HumanMessage(content=prompt)], fallback=f"{data_result}\n\n{analysis_result}")

    print(f"  [report_agent] 完成")
    return {**state, "report": report}
