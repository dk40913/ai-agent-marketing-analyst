# Multi-Agent 架構設計文件

## 系統架構

```
使用者問題
    ↓
[Supervisor Agent]  ← 負責理解問題、拆解任務、分配工作
    ↓           ↓            ↓
[Data Agent]  [Analysis Agent]  [Report Agent]
查詢銷售數據   分析異常原因      產出摘要報告
    ↓           ↓            ↓
         [Supervisor Agent]
         整合結果 → 最終回答
```

## Agent 職責邊界

| Agent | 職責 | 工具 |
|-------|------|------|
| Supervisor | 理解問題、拆解任務、分配 Worker、整合結果 | 無（純推理） |
| Data Agent | 查詢銷售與廣告數據 | query_sales_data, query_ad_spend |
| Analysis Agent | 比對數據、找出異常原因 | （接收 Data Agent 輸出） |
| Reviewer Agent | 審核計算邏輯是否符合業務常理 | 無（純審核） |
| Report Agent | 格式化輸出結構化報告 | 無（純生成） |

## 設計決策

### 為什麼加 Reviewer Agent？
行銷數據有業務常理限制（ROAS 不可能為負數、銷售額不可能一天暴增 1000%），
在 Analysis Agent 輸出後加一道審核層，防止 LLM 幻覺導致誤導性結論。

### 生產環境擴展
- 用 **RabbitMQ** 將 Agent 任務異步化，避免長任務阻塞
- 用 **Redis** 儲存 LangGraph State，支援水平擴展
