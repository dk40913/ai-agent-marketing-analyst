"""
RAG 查詢：使用者問題 → ChromaDB 搜尋相關段落 → 送 Gemini → 得到有根據的回答
"""

import os
from dotenv import load_dotenv
import chromadb

load_dotenv()
from chromadb.utils import embedding_functions
from langchain_core.messages import HumanMessage

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb")

client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_collection(name="marketing_knowledge", embedding_function=ef)

# LLM_PROVIDER 環境變數控制來源：groq（預設）或 gemini（面試 demo 用）
_provider = os.environ.get("LLM_PROVIDER", "groq").lower()

if _provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0,
    )
else:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
    )


def rag_query(question: str, n_results: int = 3) -> dict:
    """
    n_results：從 ChromaDB 撈幾段最相關的內容送給 LLM。
    太少（1段）可能資訊不夠；太多（10段）會塞爆 context window。
    3 是經驗上的合理起點。
    """

    # Step 1：向量搜尋，找最相關的段落
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # Step 2：把找到的段落拼成 context
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Step 3：送給 LLM，要求根據 context 回答
    prompt = f"""你是一位行銷數據分析師。請根據以下資料回答問題。
如果資料中沒有足夠資訊，請直接說「資料中未提及」，不要自己猜測。

【參考資料】
{context}

【問題】
{question}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
    }


if __name__ == "__main__":
    questions = [
        "山嵐咖啡線上銷售下滑的原因是什麼？",
        "城市動力的廣告表現如何？",
        "鮮橙生活有哪些風險？",
    ]
    for q in questions:
        print(f"\n問：{q}")
        result = rag_query(q)
        print(f"答：{result['answer']}")
        print(f"來源：{result['sources']}")
        print("-" * 60)
