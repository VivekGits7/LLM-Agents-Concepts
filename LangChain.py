"""
LangChain RAG Agent with OpenAI
- Takes MD files as knowledge base
- Tools: India Time, India Weather
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.documents import Document
import requests
from ddgs import DDGS

# Load environment variables
load_dotenv()

# Default document to auto-load
DEFAULT_DOCUMENT = Path(__file__).parent / "VIVEK-VISHWAKARMA-RESUME.md"


# ==================== TOOLS ====================

@tool
def get_india_time() -> str:
    """Get the current exact time in India (IST - Indian Standard Time).
    Use this tool when the user asks about the current time in India.
    """
    # IST is UTC+5:30
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
        # Using wttr.in free weather API (no API key needed)
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
    Use this tool to access GitHub repositories, LinkedIn profiles, or any web page links found in the resume.

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

        # Get text content and clean it up
        from html import unescape
        import re

        content = response.text

        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)

        # Clean up whitespace
        content = unescape(content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Limit content length
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"

        return f"Content from {url}:\n\n{content}"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


# ==================== RAG SETUP ====================

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

        # Split into chunks
        chunks = self.text_splitter.split_text(content)

        # Create documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": str(path.name), "chunk": i}
            )
            for i, chunk in enumerate(chunks)
        ]

        # Create or update vector store
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
    Use this tool to find information from the loaded knowledge base.

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


# ==================== AGENT SETUP ====================

SYSTEM_PROMPT = """You are a personal AI assistant for Vivek Vishwakarma. You have access to Vivek's resume and personal information.

The knowledge base contains Vivek Vishwakarma's resume. When users ask questions like "what is my name", "who am I", "tell me about myself", etc., you should respond based on the resume - the owner is Vivek Vishwakarma.

You have access to the following capabilities:
1. **Knowledge Base Search**: Search Vivek's resume to answer questions about his skills, experience, education, projects, and personal details.
2. **India Time**: Get the current time in India.
3. **India Weather**: Check the current weather in Indian cities.
4. **Web Search**: Search the internet for current information, news, or any topic not in the knowledge base.
5. **Fetch URL**: Access and read content from URLs/links found in the resume (GitHub repos, LinkedIn, project links).

When answering questions:
- For personal questions (name, skills, experience, education, projects), ALWAYS use the search_knowledge_base tool to find information from the resume
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


def create_rag_agent():
    """Create the RAG agent with all tools."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    tools = [get_india_time, get_india_weather, search_knowledge_base, web_search, fetch_url]

    # Create agent using LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    return agent


def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "="*60)
    print("Vivek Vishwakarma's Personal Resume Assistant")
    print("="*60)
    print("Ask me anything about Vivek's skills, experience, projects,")
    print("education, GitHub repos, or search the web!")
    print("")
    print("Commands:")
    print("  /load <file_path>  - Load additional documents")
    print("  /quit              - Exit")
    print("="*60 + "\n")

    agent = create_rag_agent()
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
                messages.append({"role": "user", "content": user_input})

                # Invoke agent with messages
                result = agent.invoke({"messages": messages})

                # Extract the last AI message
                ai_messages = [msg for msg in result["messages"] if hasattr(msg, "content") and msg.type == "ai"]
                if ai_messages:
                    response = ai_messages[-1].content
                    print(f"\nAgent: {response}")
                    messages.append({"role": "assistant", "content": response})

                # Keep history manageable
                if len(messages) > 20:
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

    parser = argparse.ArgumentParser(description="LangChain RAG Agent with OpenAI")
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
        agent = create_rag_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
        ai_messages = [msg for msg in result["messages"] if hasattr(msg, "content") and msg.type == "ai"]
        if ai_messages:
            print(f"\nAgent: {ai_messages[-1].content}")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
