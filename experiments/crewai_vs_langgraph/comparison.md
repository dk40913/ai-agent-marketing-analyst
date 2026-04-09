# CrewAI vs LangGraph 比較

同一套 Multi-Agent 邏輯（Data → Analysis → Review → Report），用兩個框架實作的差異。

---

## 核心設計哲學差異

| | LangGraph | CrewAI |
|--|-----------|--------|
| **核心抽象** | 狀態機（StateGraph） | 角色扮演團隊（Crew） |
| **你定義什麼** | Node、Edge、State | Agent（角色）、Task（任務） |
| **流程控制** | 你寫 Conditional Edge | Task 的 context 依賴關係 |
| **Supervisor** | 需要自己實作 | hierarchical process 內建 |
| **彈性** | 高（可以做循環、條件分支）| 中（sequential / hierarchical）|
| **學習曲線** | 較陡（要理解圖的概念）| 較平（直覺的角色設定）|

---

## 同一套邏輯，寫法對比

### 定義 Agent

**LangGraph**：Node = 一個 Python 函式，沒有「角色」概念
```python
def data_agent(state: dict) -> dict:
    question = state.get("question")
    response = safe_invoke_full(llm_with_tools, [HumanMessage(content=question)])
    ...
    return {**state, "data_result": data_result}
```

**CrewAI**：Agent = 有名字、角色說明、背景故事的「人物」
```python
data_analyst = Agent(
    role="行銷數據分析師",
    goal="從銷售數據找出關鍵指標和異常",
    backstory="你是一位有五年經驗的行銷數據分析師...",
    tools=[query_sales, query_ad_spend],
    llm=llm,
)
```

---

### 定義任務與依賴

**LangGraph**：Supervisor 在執行時動態決定下一步
```python
def supervisor(state):
    # 看 State 有什麼，讓 LLM 決定下一步
    next_worker = safe_invoke(llm, [HumanMessage(content=prompt)])
    return {**state, "next_worker": next_worker}

# 防禦：report 前置條件不滿足就攔截
if next_worker == "report" and (not data_result or not analysis_result):
    next_worker = "data"
```

**CrewAI**：Task 預先定義依賴，框架自動排序
```python
task_analysis = Task(
    description="根據數據摘要分析原因...",
    agent=analysis_expert,
    context=[task_data],  # 宣告「我需要 task_data 的結果」
)
task_report = Task(
    description="產出完整報告...",
    agent=report_writer,
    context=[task_data, task_analysis, task_review],  # 需要三個前置
)
```

---

### 組合與執行

**LangGraph**：手動建圖，明確定義每條 Edge
```python
graph.add_node("data_agent", data_agent)
graph.add_conditional_edges("supervisor", route_to_worker)
graph.add_edge("analysis_agent", "reviewer_agent")
graph.add_edge("reviewer_agent", "supervisor")
multi_agent = graph.compile(checkpointer=MemorySaver())

result = multi_agent.invoke({"question": q, "messages": [], ...})
```

**CrewAI**：把 agents 和 tasks 丟給 Crew，框架處理執行順序
```python
crew = Crew(
    agents=[data_analyst, analysis_expert, reviewer, report_writer],
    tasks=[task_data, task_analysis, task_review, task_report],
    process=Process.sequential,
)
result = crew.kickoff()
```

---

## 各自的適用場景

### 選 LangGraph 的情況

- 流程需要**條件分支**（問題類型不同走不同路徑）
- 需要**循環**（審核不通過就重新分析）
- 需要**跨輪對話記憶**（MemorySaver + thread_id）
- 團隊已經熟悉圖論概念，或需要精確控制每一步
- **生產環境**，需要完全掌控流程細節

### 選 CrewAI 的情況

- 快速原型，想用直覺的「角色扮演」方式描述任務
- 任務之間的依賴關係清楚、線性，不需要複雜分支
- 想利用 backstory 讓 LLM 更「入戲」，輸出風格更一致
- **探索性實驗**，不確定流程最終長什麼樣

---

## 這個專案的選擇：LangGraph

選擇 LangGraph 的理由：

1. **流程控制需求高**：Supervisor 動態決定要不要呼叫 analysis，不是所有問題都需要知識庫查詢
2. **跨輪對話記憶**：面試 demo 需要展示多輪對話能力（MemorySaver）
3. **防禦性設計**：pre-condition guard 需要在程式碼層面介入，CrewAI 的 context 依賴是宣告式的，沒有辦法在執行時動態攔截
4. **可觀察性**：LangGraph 的圖結構讓每個 Node 的輸入輸出都清楚，除錯比較容易

> CrewAI 更適合快速建立原型，LangGraph 更適合需要精確控制的生產系統。
> 這個專案兩個都實作，是為了在面試時展示「我有意識地評估過技術選項」。
