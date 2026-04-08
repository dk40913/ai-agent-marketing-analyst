"""
生成專案所需的所有假資料：
- data/csv/sales_data.csv
- data/csv/ad_spend.csv
- data/knowledge_base/brand_profiles/*.txt
- data/knowledge_base/market_reports/*.txt
- data/knowledge_base/anomaly_logs/*.txt
"""

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "csv")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

BRANDS = ["鮮橙生活", "山嵐咖啡", "城市動力"]
CHANNELS = ["線上", "線下", "App"]
PLATFORMS = ["Meta", "Google", "LINE"]
START_DATE = date(2024, 1, 1)
DAYS = 90


# ── 1. sales_data.csv ────────────────────────────────────────────────────────

def generate_sales():
    rows = []
    for brand in BRANDS:
        # 每個品牌設定不同的基準銷售額
        base = {"鮮橙生活": 80000, "山嵐咖啡": 50000, "城市動力": 65000}[brand]
        for day_offset in range(DAYS):
            d = START_DATE + timedelta(days=day_offset)
            # 週末效應
            weekend_boost = 1.3 if d.weekday() >= 5 else 1.0
            for channel in CHANNELS:
                channel_factor = {"線上": 1.2, "線下": 0.9, "App": 0.7}[channel]
                # 山嵐咖啡在第 60 天後線上銷售下滑（給 RAG 異常場景用）
                anomaly = 0.4 if (brand == "山嵐咖啡" and channel == "線上" and day_offset >= 60) else 1.0
                sales = int(base * channel_factor * weekend_boost * anomaly
                            * np.random.uniform(0.85, 1.15))
                qty = max(1, int(sales / np.random.uniform(200, 400)))
                return_rate = round(np.random.uniform(0.01, 0.08), 3)
                category = {"鮮橙生活": "日用品", "山嵐咖啡": "餐飲", "城市動力": "運動"}[brand]
                rows.append({
                    "date": d.isoformat(),
                    "brand_name": brand,
                    "category": category,
                    "channel": channel,
                    "sales_amount": sales,
                    "quantity": qty,
                    "return_rate": return_rate,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, "sales_data.csv"), index=False, encoding="utf-8-sig")
    print(f"✅ sales_data.csv — {len(df)} 筆")


# ── 2. ad_spend.csv ──────────────────────────────────────────────────────────

def generate_ad_spend():
    rows = []
    for brand in BRANDS:
        daily_budget = {"鮮橙生活": 3000, "山嵐咖啡": 2000, "城市動力": 2500}[brand]
        for day_offset in range(DAYS):
            d = START_DATE + timedelta(days=day_offset)
            for platform in PLATFORMS:
                platform_factor = {"Meta": 1.1, "Google": 1.3, "LINE": 0.8}[platform]
                spend = round(daily_budget * platform_factor * np.random.uniform(0.8, 1.2), 2)
                impressions = int(spend * np.random.uniform(80, 150))
                clicks = int(impressions * np.random.uniform(0.01, 0.05))
                conversions = int(clicks * np.random.uniform(0.03, 0.12))
                roas = round((conversions * np.random.uniform(300, 600)) / spend, 2) if spend > 0 else 0
                rows.append({
                    "date": d.isoformat(),
                    "brand_name": brand,
                    "platform": platform,
                    "spend": spend,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "roas": roas,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, "ad_spend.csv"), index=False, encoding="utf-8-sig")
    print(f"✅ ad_spend.csv — {len(df)} 筆")


# ── 3. Knowledge Base 文字檔 ──────────────────────────────────────────────────

BRAND_PROFILES = {
    "鮮橙生活": """
品牌名稱：鮮橙生活
產業類別：日用消費品（FMCG）
成立年份：2018 年
主要產品：家居清潔用品、個人護理、廚房消耗品

品牌定位：
鮮橙生活以「讓日常更輕鬆」為核心理念，主打高性價比的日用品，目標客群為 25–45 歲的都會家庭主婦及上班族。品牌強調天然成分與環保包裝，在電商平台擁有強勢表現。

通路策略：
線上通路（含官方電商與第三方平台）佔總營收約 55%，實體通路（連鎖超市、量販店）佔 35%，品牌 App 佔 10%。每年雙 11、母親節、年節是最重要的銷售旺季。

近期動態：
2024 年第一季推出「極簡系列」新品，主打無香精、無螢光劑，在社群媒體獲得正面回響。廣告投放以 Meta 與 Google 為主，LINE 官方帳號用於舊客維繫。
""".strip(),

    "山嵐咖啡": """
品牌名稱：山嵐咖啡
產業類別：餐飲（精品咖啡）
成立年份：2015 年
主要產品：精品咖啡豆、掛耳包、冷萃咖啡液、訂閱制咖啡禮盒

品牌定位：
山嵐咖啡以「台灣山林風味」為品牌故事，直接與阿里山、東部等產區農民合作，強調單一產區溯源。目標客群為 28–45 歲注重品質的咖啡愛好者，客單價偏高。

通路策略：
線上訂閱制是主要收入來源（約 50%），實體門市（台北、台中各一間旗艦店）佔 35%，App 會員預購佔 15%。口碑行銷與 KOL 合作是主要獲客方式。

近期動態：
2024 年 Q1 線上銷售出現異常下滑，內部初步判斷可能與競品大促、物流延誤及訂閱續訂率下降有關。廣告投放效益（ROAS）較去年同期下降約 18%。
""".strip(),

    "城市動力": """
品牌名稱：城市動力
產業類別：運動休閒
成立年份：2020 年
主要產品：機能運動服飾、訓練配件、運動營養品

品牌定位：
城市動力定位「城市型運動者」，主打通勤、健身、輕戶外的三合一機能需求。品牌風格年輕、設計感強，目標客群為 20–38 歲的健身愛好者與潮流族群。

通路策略：
App 自有電商是品牌重點發展管道（佔 30%），配合線上平台（40%）與實體選品店鋪（30%）。Instagram 與 YouTube 是主要行銷陣地，與健身 KOL 合作頻繁。

近期動態：
2024 年 Q1 新品「通勤系列」上市，預購反應熱烈。廣告在 Meta 與 Google 的 ROAS 表現穩定，是三個品牌中廣告效益最佳者。
""".strip(),
}

MARKET_REPORTS = {
    "鮮橙生活_Q1_2024": """
鮮橙生活 2024 Q1 市場分析報告

摘要：
2024 年第一季，鮮橙生活整體銷售額較去年同期成長 12%，主要由線上通路與新品「極簡系列」帶動。

銷售亮點：
- 線上通路成長 18%，主要受惠於平台演算法調整與開春促銷活動
- 「極簡系列」上市首月即達成首批備貨 80% 去化
- App 通路用戶月活提升 22%，回購率達 41%

廣告投放分析：
- Meta 廣告 ROAS 平均 4.2，優於行業均值 3.5
- Google 搜尋廣告點擊率（CTR）提升至 3.8%，關鍵字「天然清潔劑」排名提升
- LINE 官方帳號開封率 35%，優惠券核銷率 28%

風險提示：
原物料（天然萃取成分）價格上漲 15%，若持續可能壓縮毛利空間。競品近期亦推出類似「無添加」系列，需持續監控市佔率變化。
""".strip(),

    "山嵐咖啡_Q1_2024": """
山嵐咖啡 2024 Q1 市場分析報告

摘要：
2024 年第一季，山嵐咖啡整體銷售較去年同期下滑 8%，其中線上通路降幅達 23%，為主要警訊來源。

銷售異常分析：
- 線上訂閱續訂率從 Q4 的 68% 下滑至 Q1 的 51%，為近兩年低點
- 一月下旬物流合作商換約，導致平均配送天數從 2.1 天拉長至 4.3 天，收到大量客訴
- 競品「城市咖啡盒」在同期發動大幅折扣攻勢，搶奪部分價格敏感客群

廣告投放分析：
- Meta ROAS 從 Q4 的 5.1 降至 3.3，受物流負評影響轉換率下滑
- Google 品牌關鍵字搜尋量維持穩定，顯示品牌認知度未受大幅影響
- 建議暫緩加碼投放，優先修復物流體驗再擴大獲客

改善建議：
1. 立即處理物流合作商問題，恢復配送 SLA
2. 針對流失的訂閱用戶發送挽回優惠（首月折 30%）
3. 強化 KOL 合作，以內容行銷取代短期折扣戰
""".strip(),

    "城市動力_Q1_2024": """
城市動力 2024 Q1 市場分析報告

摘要：
2024 年第一季，城市動力銷售額達成年度目標的 28%，整體表現強勁，「通勤系列」新品是最大亮點。

銷售亮點：
- 新品「通勤系列」預購轉正式銷售後，首月銷售額占整體 35%
- App 自有電商月活用戶突破 5 萬，同比成長 67%
- 實體選品店鋪新增 3 個合作通路，覆蓋台北、台中、高雄

廣告投放分析：
- Meta ROAS 達 5.8，為三品牌最高，短影音廣告表現尤佳
- Google 購物廣告 CTR 4.2%，轉換率 6.1%，優於行業均值
- YouTube KOL 合作影片累計觀看 120 萬次，帶來明顯自然搜尋流量提升

展望 Q2：
夏季戶外運動季是重要機會，建議加碼戶外系列廣告投放。持續優化 App 會員體驗，目標 Q2 回購率達 45%。
""".strip(),
}

ANOMALY_LOGS = {
    "山嵐咖啡_線上銷售下滑_2024Q1": """
異常事件紀錄 #001
品牌：山嵐咖啡
事件類型：線上銷售異常下滑
發生期間：2024 年 1 月下旬 ～ 3 月底
嚴重程度：高（線上通路銷售較前期下滑 23%）

事件描述：
2024 年 1 月 22 日，後台偵測到山嵐咖啡線上訂單量較前 7 日均值下滑 35%。
經初步調查，發現以下三個可能成因：

1. 物流問題（主因）
   1 月 18 日完成物流合作商換約，新合作商倉儲系統磨合期導致出貨延誤。
   平均配送天數從 2.1 天拉長至 4.3 天，期間累計收到 312 則負面評價。
   Google 評分從 4.6 分下滑至 4.1 分。

2. 競品促銷（次因）
   競品「城市咖啡盒」於 1 月 20 日起推出「買 3 個月送 1 個月」訂閱優惠，
   定價比山嵐咖啡低 22%，吸引部分價格敏感客群。

3. 訂閱到期未續訂
   Q4 有大批用戶為雙 11 促銷首購，三個月訂閱到期後未續訂，
   本為季節性正常波動，但與物流問題疊加放大了整體降幅。

處理措施：
- 2 月中旬恢復原物流合作商，配送天數恢復正常
- 針對 312 位留下負評的用戶發送道歉信 + 下次訂購折抵 200 元
- 3 月起推出「訂閱滿 6 個月享 85 折」長期方案以提升留存

後續追蹤：
4 月訂閱續訂率回升至 59%，物流客訴歸零，整體趨勢回穩中。
""".strip(),

    "鮮橙生活_App通路異常_2024Q1": """
異常事件紀錄 #002
品牌：鮮橙生活
事件類型：App 通路轉換率驟降
發生期間：2024 年 2 月 1 日 ～ 2 月 14 日
嚴重程度：中（App 轉換率從 8.2% 降至 3.1%）

事件描述：
2 月 1 日 App 更新版本（v3.2.0）上線後，轉換率在 48 小時內腰斬。
用戶回報結帳流程新增「地址二次確認」步驟，導致購買摩擦增加。
熱圖分析顯示 61% 的用戶在該步驟放棄結帳。

處理措施：
2 月 7 日緊急回滾結帳流程，2 月 14 日轉換率恢復至 7.8%。
""".strip(),

    "城市動力_廣告投放異常_2024Q1": """
異常事件紀錄 #003
品牌：城市動力
事件類型：Meta 廣告帳號被短暫停權
發生期間：2024 年 3 月 5 日 ～ 3 月 8 日（3 天）
嚴重程度：低（時間短，損失可控）

事件描述：
Meta 廣告帳號因新創意素材涉及「醫療效果聲稱」（廣告文案含「改善體態、消除痠痛」）
觸發平台政策審查，帳號被暫停 3 天。

處理措施：
移除違規文案，改以「支撐核心肌群」等功能性描述替換。
3 月 8 日帳號恢復，重新投放後 ROAS 未受顯著影響。
""".strip(),

    "山嵐咖啡_ROAS下滑_2024Q1": """
異常事件紀錄 #004
品牌：山嵐咖啡
事件類型：廣告整體 ROAS 下滑
發生期間：2024 年 1 月 ～ 3 月（持續性）
嚴重程度：中（ROAS 從 5.1 降至 3.3）

事件描述：
受物流問題影響，網站評分下滑直接衝擊廣告轉換率。
新訪客在看到負評後跳出率提升 18%，導致廣告點擊後未能有效轉換。
此為物流事件（#001）的連帶影響，非廣告策略本身問題。

處理措施：
暫停擴量投放，維持基礎再行銷預算。物流恢復後（4 月）再重新評估加碼時機。
""".strip(),

    "鮮橙生活_退貨率異常_2024Q1": """
異常事件紀錄 #005
品牌：鮮橙生活
事件類型：特定 SKU 退貨率異常偏高
發生期間：2024 年 1 月 10 日 ～ 1 月 31 日
嚴重程度：中（單一 SKU 退貨率達 18%，正常值 3–5%）

事件描述：
「極簡洗碗精 500ml」上市後退貨率達 18%，客服回饋主要原因為：
1. 瓶蓋設計不良，使用時容易漏液（佔退貨原因 70%）
2. 香味與官網描述「無味」不符（佔退貨原因 20%）

處理措施：
1 月 31 日暫停銷售該 SKU，聯繫代工廠修正瓶蓋模具。
向已購用戶全額退款並附贈下次購買折扣碼。
修正版本預計 3 月重新上架。
""".strip(),
}


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(KB_DIR, "brand_profiles"), exist_ok=True)
    os.makedirs(os.path.join(KB_DIR, "market_reports"), exist_ok=True)
    os.makedirs(os.path.join(KB_DIR, "anomaly_logs"), exist_ok=True)

    generate_sales()
    generate_ad_spend()

    for brand, content in BRAND_PROFILES.items():
        path = os.path.join(KB_DIR, "brand_profiles", f"{brand}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ brand_profiles — {len(BRAND_PROFILES)} 份")

    for name, content in MARKET_REPORTS.items():
        path = os.path.join(KB_DIR, "market_reports", f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ market_reports — {len(MARKET_REPORTS)} 份")

    for name, content in ANOMALY_LOGS.items():
        path = os.path.join(KB_DIR, "anomaly_logs", f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ anomaly_logs — {len(ANOMALY_LOGS)} 份")

    print("\n全部資料生成完成 ✅")


if __name__ == "__main__":
    main()
