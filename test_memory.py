import sys
sys.path.insert(0, '.')
from agents.single_agent import agent

config = {"configurable": {"thread_id": "memory-test"}}

r1 = agent.invoke(
    {"question": "鮮橙生活的銷售額是多少？", "messages": [], "question_type": "", "detected_brand": "", "query_result": "", "answer": ""},
    config=config,
)
print("第一輪答：", r1["answer"])

r2 = agent.invoke(
    {"question": "它的退貨率正常嗎？", "messages": r1["messages"], "question_type": "", "detected_brand": "", "query_result": "", "answer": ""},
    config=config,
)
print("第二輪答：", r2["answer"])
