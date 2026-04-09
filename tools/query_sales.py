"""
tools/query_sales.py

@tool 裝飾器把這個函式變成 Agent 可以呼叫的工具。
LLM 靠函式名稱和 docstring 判斷「什麼情況下要用這個工具」，
靠參數型別和說明決定「要傳什麼值進來」。
"""

import os
import sys
import pandas as pd
from langchain_core.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")

VALID_BRANDS = {"鮮橙生活", "山嵐咖啡", "城市動力"}


@tool
def query_sales(brand: str = "", date_from: str = "", date_to: str = "") -> str:
    """
    查詢品牌銷售數據摘要，回傳總銷售額、總數量、平均退貨率等指標。

    brand: 品牌名稱，只能是「鮮橙生活」、「山嵐咖啡」、「城市動力」其中一個。
           空字串代表查詢全部品牌。
    date_from: 查詢起始日期，格式 YYYY-MM-DD，空字串代表不限制。
    date_to: 查詢結束日期，格式 YYYY-MM-DD，空字串代表不限制。
    """
    df = pd.read_csv(os.path.join(CSV_DIR, "sales_data.csv"))

    # 品牌篩選：有給品牌才篩，沒給就查全部
    if brand and brand in VALID_BRANDS:
        df = df[df["brand_name"] == brand]
    elif brand and brand not in VALID_BRANDS:
        return f"找不到品牌「{brand}」，有效品牌為：{', '.join(VALID_BRANDS)}"

    # 日期篩選：date 欄位轉成 datetime 才能比較大小
    if date_from or date_to:
        df["date"] = pd.to_datetime(df["date"])
        if date_from:
            df = df[df["date"] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df["date"] <= pd.to_datetime(date_to)]

    if df.empty:
        return "查無資料，請確認品牌名稱或日期區間。"

    # 計算摘要，格式化成文字讓 LLM 容易讀
    result = f"""銷售數據摘要（{brand or '全品牌'}）：
- 總銷售額：{int(df['sales_amount'].sum()):,}
- 總數量：{int(df['quantity'].sum()):,}
- 平均退貨率：{df['return_rate'].mean():.2%}
- 資料筆數：{len(df)} 筆
- 日期區間：{df['date'].min().strftime('%Y-%m-%d') if (date_from or date_to) else '全期間'} ～ {df['date'].max().strftime('%Y-%m-%d') if (date_from or date_to) else '全期間'}

各通路銷售額：
{df.groupby('channel')['sales_amount'].sum().apply(lambda x: f'{int(x):,}').to_string()}

各品類銷售額：
{df.groupby('category')['sales_amount'].sum().apply(lambda x: f'{int(x):,}').to_string()}"""

    return result
