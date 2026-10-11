# ============================================================
# agentic_hitl_backend.py
# Agentic Chatbot Backend
# Gemini + LangGraph + Tools + RAG + HITL
# ============================================================

from typing import TypedDict, Annotated, Any
import sqlite3
import math
import os
import requests

from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

# LangChain
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import tool

# Gemini
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# Tavily
from langchain_tavily import TavilySearch

# PDF / RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

DB_PATH = "faiss_db"


# ============================================================
# RAG DOCUMENT INGESTION
# ============================================================

def ingest_rag_document(file_path: str):
    """
    Load a PDF, split it into chunks and add it
    to the existing FAISS vector database.

    If FAISS does not exist yet, create a new one.
    """

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        return "No text could be extracted from the PDF."

    # --------------------------------------------------------
    # If FAISS already exists, add the new document
    # --------------------------------------------------------

    index_file = os.path.join(
        DB_PATH,
        "index.faiss"
    )

    if os.path.exists(index_file):

        vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        vector_store.add_documents(chunks)

    else:

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

    vector_store.save_local(DB_PATH)

    return (
        f"Successfully processed PDF. "
        f"Added {len(chunks)} chunks to the knowledge base."
    )


# ============================================================
# GET RAG RETRIEVER
# ============================================================

def get_retriever():
    """
    Load FAISS and return a retriever.
    """

    index_file = os.path.join(
        DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):
        return None

    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        },
    )

    return retriever


# ============================================================
# RAG TOOL
# ============================================================

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from uploaded PDF documents.

    Use this tool when the user asks questions related
    to an uploaded PDF.
    """

    retriever = get_retriever()

    if retriever is None:
        return (
            "No PDF document is currently available. "
            "Please upload a PDF first."
        )

    try:

        documents = retriever.invoke(query)

    except Exception as error:

        return (
            f"Could not search the PDF knowledge base: "
            f"{error}"
        )

    if not documents:
        return (
            "No relevant information was found "
            "in the uploaded PDF documents."
        )

    formatted_documents = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(
        formatted_documents
    )


# ============================================================
# TAVILY SEARCH TOOL
# ============================================================

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.

    Examples:
    2 + 2
    math.sqrt(16)
    10 * 5
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            allowed,
        )

        return str(result)

    except Exception as error:

        return (
            f"Calculation error: {error}"
        )


# ============================================================
# STOCK PRICE TOOL
# ============================================================

@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch the latest stock price.

    Examples:
    AAPL
    TSLA
    MSFT
    """

    symbol = symbol.upper().strip()

    api_key = os.getenv(
        "ALPHAVANTAGE_API_KEY"
    )

    if not api_key:

        return (
            "Alpha Vantage API key is missing. "
            "Please add ALPHAVANTAGE_API_KEY "
            "to your .env file."
        )

    url = (
        "https://www.alphavantage.co/query"
    )

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

    except requests.Timeout:

        return (
            "Stock price request timed out. "
            "Please try again."
        )

    except requests.RequestException as error:

        return (
            f"Could not connect to stock service: "
            f"{error}"
        )

    except ValueError:

        return (
            "Stock API returned an invalid response."
        )

    # --------------------------------------------------------
    # Alpha Vantage response
    # --------------------------------------------------------

    quote = data.get(
        "Global Quote"
    )

    if not quote:

        return (
            f"Could not retrieve stock price "
            f"for {symbol}.\n\n"
            f"API response: {data}"
        )

    price = quote.get(
        "05. price",
        "N/A"
    )

    change = quote.get(
        "09. change",
        "N/A"
    )

    change_percent = quote.get(
        "10. change percent",
        "N/A"
    )

    return (
        f"Stock: {symbol}\n"
        f"Current Price: ${price}\n"
        f"Change: {change}\n"
        f"Change Percent: {change_percent}"
    )


# ============================================================
# HUMAN-IN-THE-LOOP STOCK PURCHASE TOOL
# ============================================================

@tool
def purchase_stock(
    symbol: str,
    quantity: int
) -> dict:
    """
    Simulate purchasing a stock.

    HUMAN-IN-THE-LOOP:
    The graph pauses before the purchase
    and waits for human approval.
    """

    symbol = symbol.upper().strip()

    # --------------------------------------------------------
    # PAUSE GRAPH
    # --------------------------------------------------------

    decision = interrupt(
        {
            "type": "stock_purchase",
            "symbol": symbol,
            "quantity": quantity,
            "message": (
                f"Approve buying {quantity} "
                f"shares of {symbol}?"
            ),
        }
    )

    # --------------------------------------------------------
    # HUMAN APPROVED
    # --------------------------------------------------------

    if (
        isinstance(decision, str)
        and decision.lower().strip() == "yes"
    ):

        return {
            "status": "success",
            "message": (
                f"Purchase order placed for "
                f"{quantity} shares of {symbol}."
            ),
            "symbol": symbol,
            "quantity": quantity,
        }

    # --------------------------------------------------------
    # HUMAN REJECTED
    # --------------------------------------------------------

    return {
        "status": "cancelled",
        "message": (
            f"Purchase of {quantity} shares "
            f"of {symbol} was declined by human."
        ),
        "symbol": symbol,
        "quantity": quantity,
    }


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_current_weather(
    location: str
) -> str:
    """
    Get current weather for a city/location.
    """

    api_key = os.getenv(
        "OPENWEATHER_API_KEY"
    )

    if not api_key:

        return (
            "Weather API key is missing. "
            "Set OPENWEATHER_API_KEY "
            "in your .env file."
        )

    try:

        # ----------------------------------------------------
        # GEOCODING
        # ----------------------------------------------------

        geocoding_url = (
            "https://api.openweathermap.org/"
            "geo/1.0/direct"
        )

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )

        geo_response.raise_for_status()

        locations = geo_response.json()

        if not locations:

            return (
                f"Could not find location: "
                f"{location}"
            )

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]

        resolved_name = locations[0].get(
            "name",
            location
        )

        country = locations[0].get(
            "country",
            ""
        )

        state = locations[0].get(
            "state",
            ""
        )

        # ----------------------------------------------------
        # WEATHER API
        # ----------------------------------------------------

        weather_url = (
            "https://api.openweathermap.org/"
            "data/2.5/weather"
        )

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data[
            "main"
        ]["temp"]

        feels_like = weather_data[
            "main"
        ]["feels_like"]

        humidity = weather_data[
            "main"
        ]["humidity"]

        pressure = weather_data[
            "main"
        ]["pressure"]

        description = weather_data[
            "weather"
        ][0]["description"]

        wind_speed = weather_data.get(
            "wind",
            {}
        ).get(
            "speed",
            "N/A"
        )

        visibility_meters = weather_data.get(
            "visibility"
        )

        visibility_km = (
            round(
                visibility_meters / 1000,
                1
            )
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [
            resolved_name
        ]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(
            location_parts
        )

        return (
            f"Current weather in "
            f"{display_location}:\n"
            f"- Condition: "
            f"{description.title()}\n"
            f"- Temperature: "
            f"{temperature}°C\n"
            f"- Feels like: "
            f"{feels_like}°C\n"
            f"- Humidity: "
            f"{humidity}%\n"
            f"- Pressure: "
            f"{pressure} hPa\n"
            f"- Wind speed: "
            f"{wind_speed} m/s\n"
            f"- Visibility: "
            f"{visibility_km} km"
        )

    except requests.Timeout:

        return (
            "Weather request timed out."
        )

    except requests.HTTPError as error:

        status_code = (
            error.response.status_code
            if error.response
            else "unknown"
        )

        if status_code == 401:

            return (
                "OpenWeather API key "
                "is invalid or inactive."
            )

        return (
            f"Weather API HTTP error: "
            f"{status_code}"
        )

    except requests.RequestException as error:

        return (
            f"Could not connect to "
            f"weather service: {error}"
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ) as error:

        return (
            f"Unexpected weather API response: "
            f"{error}"
        )


# ============================================================
# TOOL LIST
# ============================================================

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]


# ============================================================
# BIND TOOLS TO GEMINI
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(
    state: ChatState
):
    """
    Main Gemini reasoning node.
    """

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic AI Chatbot "
            "built using LangGraph.\n\n"

            "You have access to the following tools:\n\n"

            "1. rag_tool\n"
            "Use rag_tool when the user asks about "
            "uploaded PDF documents.\n\n"

            "2. search_tool\n"
            "Use search_tool for current events, "
            "recent information, or web searches.\n\n"

            "3. calculator\n"
            "Use calculator for mathematical calculations.\n\n"

            "4. get_stock_price\n"
            "Use get_stock_price when the user asks "
            "for the current stock price.\n\n"

            "5. purchase_stock\n"
            "Use purchase_stock when the user wants "
            "to purchase a stock.\n"
            "This tool requires human approval.\n\n"

            "6. get_current_weather\n"
            "Use get_current_weather for current weather.\n\n"

            "Answer normal questions directly.\n"

            "After receiving a tool result, provide "
            "a clear final answer to the user.\n"

            "Do not invent information from uploaded PDFs."
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# SQLITE CHECKPOINT
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpoint = SqliteSaver(
    conn
)


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)


graph.add_edge(
    START,
    "chat_node"
)


graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# COMPILE
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpoint
)


# ============================================================
# THREAD HELPERS
# ============================================================

def get_all_threads():

    all_threads = set()

    try:

        for checkpoint_item in checkpoint.list(None):

            config = checkpoint_item.config

            configurable = config.get(
                "configurable",
                {}
            )

            thread_id = configurable.get(
                "thread_id"
            )

            if thread_id:
                all_threads.add(
                    thread_id
                )

    except Exception:

        pass

    return list(all_threads)