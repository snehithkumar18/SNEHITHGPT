import math
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from database import save_memory, search_memory
from rag import retrieve_from_rag



load_dotenv()


CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


@tool
def web_search(query: str) -> str:
    """
    Search the web for latest, real-time, or current information, news, events, or facts.
    """
    # 1. Primary: Tavily AI Search (Highest quality for AI agents)
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key and not tavily_key.startswith("tvly-YourActual"):
        try:
            tavily = TavilySearch(
                max_results=5,
                topic="general",
                search_depth="advanced"
            )
            result = tavily.invoke({"query": query})
            if isinstance(result, dict) and "error" in result:
                raise ValueError(str(result["error"]))
            if result:
                return str(result)
        except Exception as e:
            print(f"[Notice] Tavily search error: {e}. Falling back to DuckDuckGo...")

    # 2. Fallback: DuckDuckGo Search (Free, reliable backup)
    try:
        from ddgs import DDGS
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if results:
                formatted = []
                for idx, r in enumerate(results, start=1):
                    title = r.get("title", "")
                    href = r.get("href", "")
                    body = r.get("body", "")
                    formatted.append(f"[{idx}] {title}\nURL: {href}\n{body}")
                return "\n\n".join(formatted)
            return "No web search results found."
    except Exception as e:
        return f"Web search could not retrieve live results: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
    


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )




@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )



@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )





tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]