"""
tools/query_ad_spend.py

廣告數據查詢工具。結構與 query_sales.py 相同，
但多了 platform 篩選，因為廣告投放跨平台，各平台表現差異大。
"""

import os
import pandas as pd
from langchain_core.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")

VALID_BRANDS = {"鮮橙生活", "山嵐咖啡", "城市動力"}
VALID_PLATFORMS = {"Facebook", "Google", "Instagram", "LINE"}


@tool
def query_ad_spend(
    brand: str = "",
    platform: str = "",
    date_from: str = "",
    date_to: str = "",
) -> str:
    """
    查詢品牌廣告投放數據，回傳花費、ROAS、點擊率、轉換數等指標。

    brand: 品牌名稱，只能是「鮮橙生活」、「山嵐咖啡」、「城市動力」其中一個。
           空字串代表查詢全部品牌。
    platform: 廣告平台，只能是「Facebook」、「Google」、「Instagram」、「LINE」其中一個。
              空字串代表查詢全部平台。
    date_from: 查詢起始日期，格式 YYYY-MM-DD，空字串代表不限制。
    date_to: 查詢結束日期，格式 YYYY-MM-DD，空字串代表不限制。
    """
    df = pd.read_csv(os.path.join(CSV_DIR, "ad_spend.csv"))

    # 品牌篩選
    if brand and brand in VALID_BRANDS:
        df = df[df["brand_name"] == brand]
    elif brand and brand not in VALID_BRANDS:
        return f"找不到品牌「{brand}」，有效品牌為：{', '.join(VALID_BRANDS)}"

    # 平台篩選
    if platform and platform in VALID_PLATFORMS:
        df = df[df["platform"] == platform]
    elif platform and platform not in VALID_PLATFORMS:
        return f"找不到平台「{platform}」，有效平台為：{', '.join(VALID_PLATFORMS)}"

    # 日期篩選
    if date_from or date_to:
        df["date"] = pd.to_datetime(df["date"])
        if date_from:
            df = df[df["date"] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df["date"] <= pd.to_datetime(date_to)]

    if df.empty:
        return "查無資料，請確認品牌名稱、平台或日期區間。"

    # CTR（點擊率）= clicks / impressions，衡量廣告吸引力
    ctr = df["clicks"].sum() / df["impressions"].sum() if df["impressions"].sum() > 0 else 0

    result = f"""廣告數據摘要（{brand or '全品牌'} / {platform or '全平台'}）：
- 總花費：{df['spend'].sum():,.0f}
- 平均 ROAS：{df['roas'].mean():.2f}（每花 1 元獲得 {df['roas'].mean():.2f} 元營收）
- 總曝光數：{int(df['impressions'].sum()):,}
- 總點擊數：{int(df['clicks'].sum()):,}
- 點擊率（CTR）：{ctr:.2%}
- 總轉換數：{int(df['conversions'].sum()):,}
- 資料筆數：{len(df)} 筆

各平台花費：
{df.groupby('platform')['spend'].sum().apply(lambda x: f'{x:,.0f}').to_string()}

各平台平均 ROAS：
{df.groupby('platform')['roas'].mean().apply(lambda x: f'{x:.2f}').to_string()}"""

    return result
