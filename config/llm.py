"""
config/llm.py

集中管理 LLM 初始化與呼叫。
所有 Agent 都從這裡 import llm 和 safe_invoke，不需要各自宣告。

要換模型時，只改這個檔案就好。
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

_provider = os.environ.get("LLM_PROVIDER", "groq").lower()

if _provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it", temperature=0)
else:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def safe_invoke_full(llm_instance, messages):
    """
    和 safe_invoke 一樣有 retry 邏輯，但回傳完整 response 物件（而非只有 .content 字串）。
    用在 llm_with_tools.invoke() 的地方：呼叫端需要讀 response.tool_calls，
    所以不能只拿字串，必須拿整個物件。
    失敗三次後回傳 None，呼叫端需自行處理 None。
    """
    for attempt in range(3):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower()
            if is_rate_limit and attempt < 2:
                wait = 60 * (attempt + 1)
                print(f"  [safe_invoke_full] Rate limit，等待 {wait} 秒後重試（第 {attempt + 1} 次）...")
                time.sleep(wait)
            else:
                print(f"  [safe_invoke_full] 呼叫失敗：{str(e)[:80]}")
                return None
    return None


def safe_invoke(llm_instance, messages, fallback: str = "") -> str:
    """
    呼叫 LLM，遇到 rate limit（429）時自動等待 60 秒後重試。

    為什麼需要這個：
      Multi-Agent 系統一個問題會呼叫 LLM 5～7 次，密集測試很容易觸發
      API 的每分鐘請求限制（RPM）。與其讓程式直接崩潰，不如等一下再試。

    參數：
      llm_instance：要呼叫的 LLM 物件
      messages：傳給 LLM 的訊息列表
      fallback：重試三次都失敗時回傳的預設字串

    回傳：
      LLM 的回應文字，或 fallback 字串
    """
    for attempt in range(3):
        try:
            response = llm_instance.invoke(messages)
            # Gemma 4 等多模態模型有時回傳 list，需要轉成純字串
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return content
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower()
            if is_rate_limit and attempt < 2:
                wait = 60 * (attempt + 1)   # 第一次等 60 秒，第二次等 120 秒
                print(f"  [safe_invoke] Rate limit，等待 {wait} 秒後重試（第 {attempt + 1} 次）...")
                time.sleep(wait)
            else:
                print(f"  [safe_invoke] 呼叫失敗：{str(e)[:80]}")
                return fallback
    return fallback
