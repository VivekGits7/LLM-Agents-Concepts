"""
LangGraph RAG Agent with OpenAI
- Uses StateGraph for stateful agent workflow
- Takes MD files as knowledge base
- Tools: Current Date, India Time, India Weather, Web Search, Fetch URL, Knowledge Base Search
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Annotated, Literal
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import requests
from ddgs import DDGS

# Load environment variables
load_dotenv()

# Default document to auto-load
DEFAULT_DOCUMENT = Path(__file__).parent / "VIVEK-VISHWAKARMA-RESUME.md"


# ==================== STATE DEFINITION ====================

class AgentState(TypedDict):
    """State schema for the agent graph."""
    messages: Annotated[list, add_messages]


# ==================== TOOLS ====================

@tool
def get_current_date() -> str:
    """Get the current date.
    Use this tool when the user asks about today's date or the current date.
    """
    today = datetime.now()
    return f"Today's date: {today.strftime('%A, %B %d, %Y')}"


@tool
def get_india_time() -> str:
    """Get the current exact time in India (IST - Indian Standard Time).
    Use this tool when the user asks about the current time in India.
    """
    ist = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(ist)
    return f"Current time in India (IST): {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST"


@tool
def get_india_weather(city: str = "Delhi") -> str:
    """Get the current weather in a city in India.
    Use this tool when the user asks about weather in India.

    Args:
        city: The city name in India (default: Delhi). Examples: Mumbai, Bangalore, Chennai, Kolkata
    """
    try:
        url = f"https://wttr.in/{city},India?format=%C+%t+%h+%w"
        response = requests.get(url, timeout=10, headers={"User-Agent": "curl/7.68.0"})

        if response.status_code == 200:
            weather_data = response.text.strip()
            return f"Current weather in {city}, India: {weather_data}"
        else:
            return f"Unable to fetch weather for {city}. Please try again."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or any topic.
    Use this tool when you need to find up-to-date information from the internet.

    Args:
        query: The search query to look up on the web
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found for the query."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"[{i}] {result['title']}\n{result['body']}\nURL: {result['href']}"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching the web: {str(e)}"


@tool
def fetch_url(url: str) -> str:
    """Fetch and read content from a URL/link.
    Use this tool to access GitHub repositories, LinkedIn profiles, or any web page links.

    Args:
        url: The full URL to fetch (e.g., https://github.com/VivekGits7/Stripe-Payment-Integration)
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            return f"Failed to fetch URL. Status code: {response.status_code}"

        from html import unescape
        import re

        content = response.text
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = unescape(content)
        content = re.sub(r'\s+', ' ', content).strip()

        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"

        return f"Content from {url}:\n\n{content}"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


# ==================== RAG KNOWLEDGE BASE ====================

class RAGKnowledgeBase:
    """Manages the RAG knowledge base from markdown files."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_markdown(self, file_path: str) -> int:
        """Load a markdown file into the knowledge base."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        chunks = self.text_splitter.split_text(content)

        documents = [
            Document(
                page_content=chunk,
                metadata={"source": str(path.name), "chunk": i}
            )
            for i, chunk in enumerate(chunks)
        ]

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        return len(documents)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Search the knowledge base for relevant documents."""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)


# Global knowledge base instance
knowledge_base = RAGKnowledgeBase()


@tool
def search_knowledge_base(query: str) -> str:
    """Search the loaded markdown documents for relevant information.
    Use this tool to find information from the loaded knowledge base (resume, documents).

    Args:
        query: The search query to find relevant information
    """
    if knowledge_base.vector_store is None:
        return "No documents loaded in the knowledge base. Please load a markdown file first."

    docs = knowledge_base.search(query, k=4)

    if not docs:
        return "No relevant information found in the knowledge base."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        results.append(f"[{i}] From {source}:\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


# ==================== LANGGRAPH SETUP ====================

SYSTEM_PROMPT = """You are a personal AI assistant for Vivek Vishwakarma. You have access to Vivek's resume and personal information.

The knowledge base contains Vivek Vishwakarma's resume. When users ask questions like "what is my name", "who am I", "tell me about myself", etc., you should respond based on the resume - the owner is Vivek Vishwakarma.

You have access to the following capabilities:
1. **Knowledge Base Search**: Search Vivek's resume to answer questions about his skills, experience, education, projects, and personal details.
2. **Current Date**: Get today's date.
3. **India Time**: Get the current time in India.
4. **India Weather**: Check the current weather in Indian cities.
5. **Web Search**: Search the internet for current information, news, or any topic not in the knowledge base.
6. **Fetch URL**: Access and read content from URLs/links found in the resume (GitHub repos, LinkedIn, project links).

When answering questions:
- For personal questions (name, skills, experience, education, projects), ALWAYS use the search_knowledge_base tool to find information from the resume
- For date queries, use the get_current_date tool
- For time queries about India, use the get_india_time tool
- For weather queries about India, use the get_india_weather tool
- For general knowledge, current news, or topics not in the resume, use the web_search tool
- When asked about GitHub projects or links in the resume, first search the knowledge base to find the URL, then use fetch_url to get details
- Be helpful, accurate, and friendly
- Respond as if you are Vivek's personal assistant helping him or others learn about his background

Important URLs from Vivek's resume:
- GitHub: https://github.com/VivekGits7
- LinkedIn: https://www.linkedin.com/in/vivek-vishwakarma-
- Stripe Payment Integration: https://github.com/VivekGits7/Stripe-Payment-Integration
- Property Finder Chat AI: https://github.com/VivekGits7/Property-Search-Chat-AI

Remember: The resume owner is Vivek Vishwakarma from Jabalpur, Madhya Pradesh, India."""


# All tools for the agent
tools = [get_current_date, get_india_time, get_india_weather, search_knowledge_base, web_search, fetch_url]


def create_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Create the LangGraph agent workflow."""

    # Initialize the LLM with tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)

    # Define the agent node
    def agent_node(state: AgentState) -> AgentState:
        """The agent node that decides what to do next."""
        messages = state["messages"]

        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Define the routing function
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM made tool calls, route to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Otherwise, end the conversation turn
        return "__end__"

    # Create the tool node using prebuilt ToolNode
    tool_node = ToolNode(tools)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")  # After tools, go back to agent

    # Compile and return the graph
    return graph.compile()


# ==================== INTERACTIVE MODE ====================

def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("LangGraph RAG Agent - Vivek's Personal Assistant")
    print("=" * 60)
    print("Ask me anything about Vivek's skills, experience, projects,")
    print("education, GitHub repos, or search the web!")
    print("")
    print("Commands:")
    print("  /load <file_path>  - Load additional documents")
    print("  /quit              - Exit")
    print("=" * 60 + "\n")

    # Create the graph
    app = create_graph()

    # Maintain conversation history
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            elif user_input.startswith("/load "):
                file_path = user_input[6:].strip()
                try:
                    num_chunks = knowledge_base.load_markdown(file_path)
                    print(f"Loaded {num_chunks} chunks from '{file_path}'")
                except Exception as e:
                    print(f"Error loading file: {e}")

            else:
                # Add user message to history
                messages.append(HumanMessage(content=user_input))

                # Invoke the graph with current state
                result = app.invoke({"messages": messages})

                # Extract the final AI response
                final_messages = result["messages"]
                ai_response = None

                # Find the last AI message (not a tool message)
                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        ai_response = msg.content
                        break

                if ai_response:
                    print(f"\nAgent: {ai_response}")
                    # Update messages with the full conversation
                    messages = final_messages

                # Keep history manageable (last 20 messages)
                if len(messages) > 20:
                    # Keep system message if present, then last 19 messages
                    if isinstance(messages[0], SystemMessage):
                        messages = [messages[0]] + messages[-19:]
                    else:
                        messages = messages[-20:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def auto_load_default_document():
    """Auto-load the default document on startup."""
    if DEFAULT_DOCUMENT.exists():
        try:
            num_chunks = knowledge_base.load_markdown(str(DEFAULT_DOCUMENT))
            print(f"Auto-loaded {num_chunks} chunks from '{DEFAULT_DOCUMENT.name}'")
            return True
        except Exception as e:
            print(f"Warning: Could not auto-load default document: {e}")
            return False
    else:
        print(f"Note: Default document not found at {DEFAULT_DOCUMENT}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph RAG Agent with OpenAI")
    parser.add_argument("--load", type=str, help="Load a markdown file at startup")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--no-auto-load", action="store_true", help="Skip auto-loading default document")

    args = parser.parse_args()

    # Auto-load default document unless disabled
    if not args.no_auto_load:
        auto_load_default_document()

    # Load additional file if specified
    if args.load:
        try:
            num_chunks = knowledge_base.load_markdown(args.load)
            print(f"Loaded {num_chunks} chunks from '{args.load}'")
        except Exception as e:
            print(f"Error loading file: {e}")
            return

    if args.query:
        # Single query mode
        app = create_graph()
        result = app.invoke({"messages": [HumanMessage(content=args.query)]})

        # Find the final AI response
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                print(f"\nAgent: {msg.content}")
                break
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
