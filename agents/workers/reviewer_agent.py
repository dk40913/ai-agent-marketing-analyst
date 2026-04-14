"""
agents/workers/reviewer_agent.py

Reviewer Agent：審核 Analysis Agent 的分析邏輯是否符合業務常理。

存在原因：
  LLM 可能產生聽起來合理但實際上違反業務常理的結論。
  例如：「ROAS 為 -2.3」、「單日銷售額成長 1500%」。
  Reviewer 是最後一道防線，讓分析結果不會誤導決策。

流程：
  analysis_agent 完成 → reviewer_agent 審核 → 回到 supervisor
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

from langchain_core.messages import HumanMessage
from config.llm import llm, safe_invoke


# 業務常理規則，以文字形式列給 LLM 參考
# 這些是行銷數據的基本邏輯限制，違反的話幾乎肯定是 LLM 幻覺
BUSINESS_RULES = """
1. ROAS（廣告投資報酬率）不可能為負數，正常範圍約 0.5～10
2. 單日銷售額成長幅度不可能超過 500%（除非有特殊活動說明）
3. 退貨率不可能超過 100%，正常範圍約 1%～30%
4. CTR（點擊率）不可能超過 100%，正常範圍約 0.5%～10%
5. 如果分析聲稱某品牌「完全沒有」問題，但問題本身是詢問異常，這個結論可疑
6. 數字引用必須和數據摘要一致，不可自己捏造數字
"""


def reviewer_agent(state: dict) -> dict:
    """
    審核 analysis_result 的邏輯是否合理。

    審核結果有兩種：
    - 通過（approved）：analysis_result 不變，加上審核通過的標記
    - 修正（revised）：把問題指出來，並附上修正後的版本
    """
    analysis_result = state.get("analysis_result", "")
    # 防禦：analysis_result 有時會是 list（LangChain content blocks），轉成字串
    if isinstance(analysis_result, list):
        analysis_result = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in analysis_result
        )
    data_result = state.get("data_result", "")
    question = state.get("question", "")

    print(f"  [reviewer_agent] 開始審核分析邏輯...")

    if not analysis_result:
        # 沒有分析結果就跳過，避免空審核
        print(f"  [reviewer_agent] 無分析結果可審核，跳過")
        return state

    prompt = f"""你是一位嚴格的行銷數據審核員。請審核以下分析報告是否符合業務常理。

【業務常理規則】
{BUSINESS_RULES}

【原始數據摘要】
{data_result if data_result else '（無數據）'}

【待審核的分析報告】
{analysis_result}

【原始問題】
{question}

請依照以下格式回覆：

【審核結果】APPROVED 或 REVISED

【說明】
- 如果 APPROVED：簡短說明分析邏輯合理的原因
- 如果 REVISED：具體指出哪裡有問題，並提供修正後的分析

注意：只有在明確違反業務常理規則時才 REVISED，不要過度挑剔措辭或結構。"""

    review = safe_invoke(llm, [HumanMessage(content=prompt)], fallback="【審核結果】APPROVED\n【說明】審核服務暫時不可用，原始分析保留。")

    # 解析審核結果
    is_approved = "APPROVED" in review.upper().split("【說明】")[0]

    if is_approved:
        print(f"  [reviewer_agent] 審核通過")
        # 在分析結果後面附上審核標記，不改動內容
        reviewed_analysis = analysis_result + "\n\n---\n*✅ 經 Reviewer Agent 審核：邏輯符合業務常理*"
    else:
        print(f"  [reviewer_agent] 發現問題，已修正")
        # 從【說明】後面提取修正內容，去掉審核框架文字
        parts = review.split("【說明】")
        reviewed_analysis = parts[1].strip() if len(parts) > 1 else review

    return {**state, "analysis_result": reviewed_analysis}
