"""
FastAPI 資料查詢服務
提供三個端點供 Agent 呼叫：
  GET /brands              → 品牌列表
  GET /sales               → 銷售數據查詢
  GET /adspend             → 廣告投放查詢
"""

import os
import sys
import uuid
import asyncio
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

BASE_DIR_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR_ROOT)

from agents.supervisor import multi_agent

app = FastAPI(title="行銷數據 API", version="1.0")

BASE_DIR = BASE_DIR_ROOT
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")


def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(CSV_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"{filename} 尚未生成，請先執行 data/generate_data.py")
    return pd.read_csv(path)


# ── /brands ───────────────────────────────────────────────────────────────────

@app.get("/brands", summary="取得所有品牌列表")
def get_brands():
    df = load_csv("sales_data.csv")
    brands = df["brand_name"].unique().tolist()
    return {"brands": brands}


# ── /sales ────────────────────────────────────────────────────────────────────

@app.get("/sales", summary="查詢銷售數據")
def get_sales(
    brand: Optional[str] = Query(None, description="品牌名稱，不填則回傳所有品牌"),
    start: Optional[str] = Query(None, description="開始日期，格式 YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="結束日期，格式 YYYY-MM-DD"),
    channel: Optional[str] = Query(None, description="通路：線上 / 線下 / App"),
):
    df = load_csv("sales_data.csv")

    if brand:
        df = df[df["brand_name"] == brand]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"找不到品牌：{brand}")
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    if channel:
        df = df[df["channel"] == channel]

    summary = {
        "total_sales": int(df["sales_amount"].sum()),
        "total_quantity": int(df["quantity"].sum()),
        "avg_return_rate": round(df["return_rate"].mean(), 4) if not df.empty else 0,
        "record_count": len(df),
    }

    return {
        "filters": {"brand": brand, "start": start, "end": end, "channel": channel},
        "summary": summary,
        "records": df.to_dict(orient="records"),
    }


# ── /adspend ──────────────────────────────────────────────────────────────────

@app.get("/adspend", summary="查詢廣告投放數據")
def get_adspend(
    brand: Optional[str] = Query(None, description="品牌名稱，不填則回傳所有品牌"),
    platform: Optional[str] = Query(None, description="平台：Meta / Google / LINE"),
    start: Optional[str] = Query(None, description="開始日期，格式 YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="結束日期，格式 YYYY-MM-DD"),
):
    df = load_csv("ad_spend.csv")

    if brand:
        df = df[df["brand_name"] == brand]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"找不到品牌：{brand}")
    if platform:
        df = df[df["platform"] == platform]
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    summary = {
        "total_spend": round(df["spend"].sum(), 2),
        "total_conversions": int(df["conversions"].sum()),
        "avg_roas": round(df["roas"].mean(), 2) if not df.empty else 0,
        "avg_ctr": round((df["clicks"].sum() / df["impressions"].sum() * 100), 2) if df["impressions"].sum() > 0 else 0,
        "record_count": len(df),
    }

    return {
        "filters": {"brand": brand, "platform": platform, "start": start, "end": end},
        "summary": summary,
        "records": df.to_dict(orient="records"),
    }


# ── /analyze ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None  # 不填代表開新對話，填入則繼續舊對話


@app.post("/analyze", summary="Multi-Agent 分析（輸入問題，回傳完整分析報告）")
async def analyze(req: AnalyzeRequest):
    """
    把使用者的自然語言問題丟給 Multi-Agent 系統。
    流程：Supervisor → Data / Analysis / Reviewer / Report Agent → 最終回答

    多輪對話用法：
      第一次不帶 thread_id → 系統產生新的，回傳給前端
      之後每次帶入同一個 thread_id → MemorySaver 自動載入歷史 messages
    """
    # 有帶 thread_id 就繼續舊對話，沒有就開新的
    thread_id = req.thread_id or str(uuid.uuid4())
    is_new_conversation = req.thread_id is None

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": req.question,
        # 新對話才清空 messages；舊對話讓 MemorySaver 自動從記憶體載入歷史
        "messages": [] if is_new_conversation else None,
        "data_result": "",
        "analysis_result": "",
        "report": "",
        "next_worker": "",
        "final_answer": "",
    }

    # None 的欄位不傳入，避免覆蓋 MemorySaver 已儲存的值
    if not is_new_conversation:
        del initial_state["messages"]

    result = await asyncio.to_thread(multi_agent.invoke, initial_state, config)

    return {
        "question": req.question,
        "answer": result["final_answer"],
        "thread_id": thread_id,
        "is_new_conversation": is_new_conversation,
    }
