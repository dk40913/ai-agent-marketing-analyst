"""
Multi-Agent 行銷分析助手 Demo

展示多輪對話能力：AI 能記住對話歷史，理解「那」、「它」等代詞的上下文。

使用方式：
  1. 先啟動 FastAPI server：uvicorn api.main:app --port 8000
  2. 另開一個 terminal 執行：python demo.py
"""

import requests
import textwrap
import sys

BASE_URL = "http://localhost:8000"
WIDTH = 70  # 輸出換行寬度


def check_server():
    """確認 FastAPI server 有在跑，否則提示使用者先啟動。"""
    try:
        requests.get(f"{BASE_URL}/brands", timeout=2)
    except requests.exceptions.ConnectionError:
        print("❌ 找不到 FastAPI server，請先執行：")
        print("   uvicorn api.main:app --port 8000")
        sys.exit(1)


def format_answer(text: str) -> str:
    """把 AI 回答排版成易讀的格式，每行最多 WIDTH 個字元。"""
    lines = text.strip().split("\n")
    formatted = []
    for line in lines:
        if len(line) <= WIDTH:
            formatted.append(line)
        else:
            # 長行自動換行，縮排對齊
            wrapped = textwrap.fill(line, width=WIDTH, subsequent_indent="    ")
            formatted.append(wrapped)
    return "\n".join(formatted)


def main():
    check_server()

    print("=" * WIDTH)
    print("  行銷數據分析助手")
    print("  輸入問題開始對話，輸入 quit 或按 Ctrl+C 結束")
    print("  輸入 new 開始新的對話（清除記憶）")
    print("=" * WIDTH)

    thread_id = None  # 第一次沒有 thread_id，開新對話

    while True:
        try:
            question = input("\n你：").strip()
        except KeyboardInterrupt:
            print("\n\n再見！")
            break

        if not question:
            continue

        if question.lower() == "quit":
            print("再見！")
            break

        if question.lower() == "new":
            thread_id = None
            print("── 開始新對話，記憶已清除 ──")
            continue

        # 帶著 thread_id 發送 request
        # 第一次 thread_id 是 None，server 會產生新的
        # 之後每次都帶同一個 thread_id，MemorySaver 會載入歷史
        body = {"question": question}
        if thread_id:
            body["thread_id"] = thread_id

        try:
            print("\nAI：（分析中...）", end="\r")
            resp = requests.post(
                f"{BASE_URL}/analyze",
                json=body,
            ).json()
        except requests.exceptions.Timeout:
            print("AI：（回應逾時，請重試）")
            continue
        except Exception as e:
            print(f"AI：（發生錯誤：{e}）")
            continue

        # 存下 thread_id，下次繼續同一個對話
        thread_id = resp.get("thread_id")
        answer = resp.get("answer", "（無回應）")

        print(f"\nAI：{format_answer(answer)}")

        # 顯示 thread_id 讓使用者知道目前在哪個對話
        is_new = resp.get("is_new_conversation", True)
        status = "新對話" if is_new else "繼續對話"
        print(f"\n[{status} | thread: {thread_id[:8]}...]")
        print("-" * WIDTH)


if __name__ == "__main__":
    main()
