# RAG Agent Project

A comprehensive RAG (Retrieval-Augmented Generation) project demonstrating different approaches to building AI agents using LangChain, LangGraph, and OpenAI.

---

## Table of Contents

- [Concepts Overview](#concepts-overview)
- [Project Structure](#project-structure)
- [Python Files Explained](#python-files-explained)
- [Setup with UV](#setup-with-uv)
- [Running the Project](#running-the-project)

---

## Concepts Overview

### What is LangChain?

**LangChain** is a framework for developing applications powered by large language models (LLMs). It provides:

- **Chains**: Sequential pipelines that combine multiple LLM calls and tools
- **Tools**: Functions that agents can use to interact with the outside world (APIs, databases, etc.)
- **Embeddings**: Convert text to vector representations for semantic search
- **Vector Stores**: Store and retrieve embeddings efficiently (FAISS, ChromaDB, etc.)
- **Document Loaders**: Load data from various sources (PDFs, markdown, web pages)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangChain Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │  Prompt  │───>│   LLM    │───>│  Output  │───>│  Parser  │      │
│   │ Template │    │ (OpenAI) │    │          │    │          │      │
│   └──────────┘    └────┬─────┘    └──────────┘    └──────────┘      │
│                        │                                            │
│                        v                                            │
│              ┌─────────────────┐                                    │
│              │     Tools       │                                    │
│              ├─────────────────┤                                    │
│              │ - Web Search    │                                    │
│              │ - Weather API   │                                    │
│              │ - Calculator    │                                    │
│              │ - Vector Store  │                                    │
│              └─────────────────┘                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Use Case**: When you need a straightforward agent with tool-calling capabilities and don't require complex state management.

---

### What is LangGraph?

**LangGraph** is a library built on top of LangChain for creating **stateful, multi-step agent workflows** using a graph-based approach.

Key features:
- **StateGraph**: Define agent state that persists across conversation turns
- **Nodes**: Individual processing steps (agent reasoning, tool execution)
- **Edges**: Control flow between nodes (conditional routing, loops)
- **Cycles**: Support for iterative agent loops (agent -> tools -> agent)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangGraph Workflow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                         ┌─────────────┐                             │
│                         │   START     │                             │
│                         └──────┬──────┘                             │
│                                │                                    │
│                                v                                    │
│                    ┌───────────────────────┐                        │
│                    │      Agent Node       │<───────┐               │
│                    │  (LLM + Reasoning)    │        │               │
│                    └───────────┬───────────┘        │               │
│                                │                    │               │
│                    ┌───────────v───────────┐        │               │
│                    │   Has Tool Calls?     │        │               │
│                    └───────────┬───────────┘        │               │
│                         │             │             │               │
│                    Yes  │             │  No         │               │
│                         v             v             │               │
│              ┌─────────────────┐  ┌───────┐         │               │
│              │   Tools Node    │  │  END  │         │               │
│              │ (Execute Tools) │  └───────┘         │               │
│              └────────┬────────┘                    │               │
│                       │                             │               │
│                       └─────────────────────────────┘               │
│                              (Loop back)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**State Management**:
```
┌─────────────────────────────────────┐
│            AgentState               │
├─────────────────────────────────────┤
│  messages: [                        │
│    HumanMessage("What's the time?") │
│    AIMessage(tool_calls=[...])      │
│    ToolMessage("10:30 AM IST")      │
│    AIMessage("It's 10:30 AM IST")   │
│  ]                                  │
└─────────────────────────────────────┘
```

**Use Case**: When you need complex agent workflows with state management, conditional logic, or multi-step reasoning.

---

### What is RAG (Retrieval-Augmented Generation)?

**RAG** is a technique that enhances LLM responses by retrieving relevant information from a knowledge base before generating answers.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                    1. INDEXING PHASE                          ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ Document │───>│  Chunk   │───>│ Embedding│───>│ Vector Store │   │
│  │ (MD/PDF) │    │  Split   │    │  Model   │    │ (FAISS/Chroma│   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘   │
│                                                                     │
│       "Resume content..."   ["Chunk 1",    [0.1, 0.3,      ┌────┐   │
│                              "Chunk 2"]     0.8, ...]      │ DB │   │
│                                                            └────┘   │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                    2. RETRIEVAL PHASE                         ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     ┌──────────────┐  │
│  │  Query   │───>│ Embedding│───>│Similarity│───> │ Top K Chunks │  │
│  │          │    │  Model   │    │  Search  │     │              │  │
│  └──────────┘    └──────────┘    └──────────┘     └──────────────┘  │
│                                                                     │
│   "What skills      [0.2, 0.4,     Find closest    ["Python...",    │
│    does he have?"    0.7, ...]     vectors         "AI/ML..."]      │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                   3. GENERATION PHASE                         ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         LLM Prompt                           │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  Context: [Retrieved Chunks]                                 │   │
│  │  Question: "What skills does he have?"                       │   │
│  │  -> Answer based on context                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               v                                     │
│                    ┌────────────────────┐                           │
│                    │  Generated Answer  │                           │
│                    │  "He has Python,   │                           │
│                    │   JavaScript..."   │                           │
│                    └────────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Reduces hallucinations by grounding responses in real data
- Allows LLMs to access private/updated information not in training data
- More cost-effective than fine-tuning

---

### What is Multi-Agent?

**Multi-Agent systems** involve multiple specialized AI agents working together to accomplish complex tasks.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Multi-Agent Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    SUPERVISOR PATTERN                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│                      ┌──────────────────┐                           │
│                      │ Supervisor Agent │                           │
│                      │  (Orchestrator)  │                           │
│                      └────────┬─────────┘                           │
│                               │                                     │
│              ┌────────────────┼────────────────┐                    │
│              │                │                │                    │
│              v                v                v                    │
│     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│     │   Research   │ │    Coder     │ │   Reviewer   │              │
│     │    Agent     │ │    Agent     │ │    Agent     │              │
│     └──────────────┘ └──────────────┘ └──────────────┘              │
│           │                │                │                       │
│           v                v                v                       │
│     ┌──────────┐     ┌──────────┐     ┌──────────┐                  │
│     │Web Search│     │Write Code│     │Review &  │                  │
│     │Read Docs │     │Run Tests │     │Feedback  │                  │
│     └──────────┘     └──────────┘     └──────────┘                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   COLLABORATIVE PATTERN                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│     ┌─────────┐        ┌─────────┐        ┌─────────┐               │
│     │ Agent A │<──────>│ Agent B │<──────>│ Agent C │               │
│     │(Planner)│        │(Executor│        │(Checker)│               │
│     └─────────┘        └─────────┘        └─────────┘               │
│          │                  │                  │                    │
│          └──────────────────┴──────────────────┘                    │
│                    Shared State / Memory                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Types**:
- **Supervisor Agent**: One agent coordinates and delegates tasks to specialist agents
- **Collaborative Agents**: Agents work together, passing information between each other
- **Hierarchical Agents**: Agents organized in layers with different responsibilities

**Use Case**: Complex tasks that benefit from specialization (e.g., research agent + coding agent + review agent).

---

## Project Structure

```
Rag/
├── LangChain.py          # LangChain-based RAG agent
├── LangGraph.py          # LangGraph-based RAG agent with state management
├── RAG-Agent.py          # Pure OpenAI RAG agent with ChromaDB
├── pyproject.toml        # UV project configuration
├── uv.lock               # UV lock file (auto-generated)
├── .env                  # Environment variables (OPENAI_API_KEY)
├── chroma_db/            # ChromaDB persistence directory
└── VIVEK-VISHWAKARMA-RESUME.md  # Sample knowledge base document
```

---

## Python Files Explained

### 1. LangChain.py - LangChain RAG Agent

This file implements a RAG agent using LangChain's `create_agent` function.

**Architecture Flow**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                      LangChain Agent Flow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐    │
│  │  User asks  │────>│  Agent +    │────>│  Tool Selection     │    │
│  │  question   │     │  System     │     │  (which tool to use)│    │
│  └─────────────┘     │  Prompt     │     └──────────┬──────────┘    │
│                      └─────────────┘                │               │
│                                                     v               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐    │
│  │  Response   │<────│  Generate   │<────│   Execute Tool      │    │
│  │  to User    │     │  Answer     │     │   (get results)     │    │
│  └─────────────┘     └─────────────┘     └─────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 1: Define Tools with `@tool` Decorator

Tools are functions the agent can call. Each tool has a description that helps the agent decide when to use it:

```python
from langchain_core.tools import tool

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
    """
    url = f"https://wttr.in/{city},India?format=%C+%t+%h+%w"
    response = requests.get(url, timeout=10)
    return f"Current weather in {city}, India: {response.text.strip()}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search the loaded markdown documents for relevant information.
    Use this tool to find information from the loaded knowledge base.
    """
    docs = knowledge_base.search(query, k=4)
    # Return formatted results...
```

#### Step 2: RAG Knowledge Base Class

This class handles document loading, chunking, and vector storage:

```python
class RAGKnowledgeBase:
    def __init__(self):
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_markdown(self, file_path: str) -> int:
        """Load a markdown file into the knowledge base."""
        content = Path(file_path).read_text(encoding="utf-8")
        chunks = self.text_splitter.split_text(content)

        # Create documents with metadata
        documents = [
            Document(page_content=chunk, metadata={"source": file_path})
            for chunk in chunks
        ]

        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return len(documents)
```

#### Step 3: Create the Agent

```python
def create_rag_agent():
    """Create the RAG agent with all tools."""

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # List of tools the agent can use
    tools = [get_india_time, get_india_weather, search_knowledge_base, web_search, fetch_url]

    # Create agent with LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    return agent
```

#### Step 4: Interactive Loop

```python
def interactive_mode():
    agent = create_rag_agent()
    messages = []

    while True:
        user_input = input("You: ").strip()

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Invoke agent with messages
        result = agent.invoke({"messages": messages})

        # Extract and display the response
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        if ai_messages:
            print(f"Agent: {ai_messages[-1].content}")
```

---

### 2. LangGraph.py - LangGraph RAG Agent

This file implements a stateful RAG agent using LangGraph's `StateGraph` for explicit workflow control.

**Architecture Flow**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent Workflow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│     ┌─────────┐                                                     │
│     │  START  │                                                     │
│     └────┬────┘                                                     │
│          │                                                          │
│          v                                                          │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                     AGENT NODE                            │      │
│  │  - Receives messages (including history)                  │      │
│  │  - LLM decides: answer directly OR call tools             │      │
│  │  - If tools needed: adds tool_calls to response           │      │
│  └───────────────────────────┬───────────────────────────────┘      │
│                              │                                      │
│                   ┌──────────┴──────────┐                           │
│                   │  should_continue()  │                           │
│                   │  Check: tool_calls? │                           │
│                   └──────────┬──────────┘                           │
│                              │                                      │
│              ┌───────────────┴───────────────┐                      │
│              │                               │                      │
│         has tools                       no tools                    │
│              │                               │                      │
│              v                               v                      │
│  ┌─────────────────────┐            ┌─────────────┐                 │
│  │     TOOLS NODE      │            │     END     │                 │
│  │  - Execute tools    │            │  (Response  │                 │
│  │  - Return results   │            │   ready)    │                 │
│  └──────────┬──────────┘            └─────────────┘                 │
│             │                                                       │
│             └───────────────────┐                                   │
│                                 │                                   │
│                        Loop back to AGENT                           │
│                        (with tool results)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 1: Define State Schema

The state holds conversation messages that persist across the workflow:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State schema for the agent graph."""
    messages: Annotated[list, add_messages]  # add_messages is a reducer
```

#### Step 2: Define Graph Nodes

**Agent Node** - LLM reasoning and tool selection:

```python
def agent_node(state: AgentState) -> AgentState:
    """The agent node that decides what to do next."""
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Call LLM with tools
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

**Conditional Router** - Decides next step:

```python
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made tool calls, route to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end the conversation turn
    return "__end__"
```

#### Step 3: Build the Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def create_graph():
    """Create the LangGraph agent workflow."""

    # Initialize LLM with tools bound
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)

    # Create tool node using prebuilt ToolNode
    tool_node = ToolNode(tools)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "agent")                    # START -> agent
    graph.add_conditional_edges(                       # agent -> tools OR end
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")                  # tools -> agent (loop)

    # Compile and return
    return graph.compile()
```

#### Step 4: Interactive Loop with State

```python
def interactive_mode():
    app = create_graph()
    messages = []

    while True:
        user_input = input("You: ").strip()

        # Add user message
        messages.append(HumanMessage(content=user_input))

        # Invoke the graph with current state
        result = app.invoke({"messages": messages})

        # Find the final AI response (not tool response)
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                print(f"Agent: {msg.content}")
                break

        # Update messages with full conversation
        messages = result["messages"]
```

---

### 3. RAG-Agent.py - Pure OpenAI RAG Agent

This file implements a simpler RAG system using OpenAI directly with ChromaDB for persistent storage.

**Architecture Flow**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Pure RAG Pipeline (No Tools)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    DOCUMENT INGESTION                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│    ┌──────────┐    ┌────────────┐    ┌───────────────────────┐      │
│    │ Markdown │───>│   Chunk    │───>│  ChromaDB + OpenAI    │      │
│    │   File   │    │   (1000    │    │  Embeddings           │      │
│    │          │    │   chars)   │    │  (Auto-embedded)      │      │
│    └──────────┘    └────────────┘    └───────────────────────┘      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      QUERY PHASE                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│    ┌──────────┐    ┌────────────┐    ┌───────────────────────┐      │
│    │  User    │───>│  ChromaDB  │───>│   Top 5 Similar       │      │
│    │  Query   │    │  Search    │    │   Chunks Retrieved    │      │
│    └──────────┘    └────────────┘    └───────────┬───────────┘      │
│                                                  │                  │
│                                                  v                  │
│    ┌──────────┐    ┌────────────┐    ┌───────────────────────┐      │
│    │ Response │<───│  OpenAI    │<───│   Context + Question  │      │
│    │          │    │  Chat API  │    │   Prompt              │      │
│    └──────────┘    └────────────┘    └───────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 1: Text Chunking Function

Smart chunking that tries to break at natural boundaries:

```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for better context preservation."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            newline_pos = text.rfind('\n\n', start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 2
            else:
                # Look for sentence break
                for punct in ['. ', '! ', '? ', '\n']:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos > start + chunk_size // 2:
                        end = punct_pos + len(punct)
                        break

        chunks.append(text[start:end].strip())
        start = end - overlap  # Overlap for context continuity

    return chunks
```

#### Step 2: RAGAgent Class with ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

class RAGAgent:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Create OpenAI embedding function
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="markdown_rag",
            embedding_function=self.embedding_fn
        )
```

#### Step 3: Add Documents

```python
def add_document(self, file_path: str) -> int:
    """Add a markdown document to the RAG database."""
    content = Path(file_path).read_text(encoding='utf-8')
    chunks = chunk_text(content)

    # Generate unique IDs
    file_name = Path(file_path).name
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

    # Metadata for each chunk
    metadatas = [
        {"source": file_path, "file_name": file_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # Add to collection (ChromaDB auto-embeds)
    self.collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)
```

#### Step 4: Query with RAG

```python
def query(self, question: str, n_context: int = 5) -> str:
    """Query the RAG agent with a question."""

    # Retrieve relevant context
    results = self.collection.query(
        query_texts=[question],
        n_results=n_context,
        include=["documents", "metadatas"]
    )

    # Build context string
    context_parts = []
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i].get('file_name', 'Unknown')
        context_parts.append(f"[Source {i+1}: {source}]\n{doc}")

    context_str = "\n\n---\n\n".join(context_parts)

    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on the provided context only."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content
```

---

## Setup with UV

### Step 1: Install UV

UV is a fast Python package manager written in Rust.

**Windows (PowerShell)**:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Or using pip**:
```bash
pip install uv
```

### Step 2: Clone/Navigate to Project

```bash
cd path/to/Rag
```

### Step 3: Sync Dependencies

This command installs all dependencies from `pyproject.toml`:

```bash
uv sync
```

UV will:
1. Create a virtual environment (`.venv/`)
2. Install all dependencies listed in `pyproject.toml`
3. Generate/update `uv.lock` for reproducible builds

### Step 4: Setup Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

---

## Running the Project

### Running LangChain Agent

**Interactive Mode** (default):
```bash
uv run python LangChain.py
```

**Single Query Mode**:
```bash
uv run python LangChain.py --query "What are Vivek's skills?"
```

**Load Additional Document**:
```bash
uv run python LangChain.py --load path/to/document.md
```

**Skip Auto-Loading Default Document**:
```bash
uv run python LangChain.py --no-auto-load
```

---

### Running LangGraph Agent

**Interactive Mode** (default):
```bash
uv run python LangGraph.py
```

**Single Query Mode**:
```bash
uv run python LangGraph.py --query "Tell me about Vivek's projects"
```

**With Additional Document**:
```bash
uv run python LangGraph.py --load another-doc.md
```

---

### Running RAG-Agent (ChromaDB)

**Interactive Mode** (default):
```bash
uv run python RAG-Agent.py
```

**Or explicitly**:
```bash
uv run python RAG-Agent.py --interactive
```

**Add Document and Query**:
```bash
uv run python RAG-Agent.py --add VIVEK-VISHWAKARMA-RESUME.md --query "What is Vivek's experience?"
```

**Clear Knowledge Base**:
```bash
uv run python RAG-Agent.py --clear
```

---

## Interactive Commands

Once running in interactive mode, use these commands:

| Command | Description |
|---------|-------------|
| `/load <file>` | Load a markdown file into knowledge base |
| `/add <file>` | Add document (RAG-Agent.py only) |
| `/search <query>` | Search documents (RAG-Agent.py only) |
| `/sources` | List all sources (RAG-Agent.py only) |
| `/clear` | Clear knowledge base (RAG-Agent.py only) |
| `/quit` | Exit the application |

---

## Comparison Table

| Feature | LangChain.py | LangGraph.py | RAG-Agent.py |
|---------|--------------|--------------|--------------|
| Framework | LangChain | LangGraph | Pure OpenAI |
| State Management | Basic | Advanced (StateGraph) | None |
| Vector Store | FAISS | FAISS | ChromaDB |
| Persistence | In-memory | In-memory | Persistent |
| Tool Support | Yes (5 tools) | Yes (5 tools) | No (RAG only) |
| Web Search | Yes | Yes | No |
| Complexity | Medium | High | Low |
| Best For | Simple agents | Complex workflows | Basic RAG |

---

## Dependencies

All managed via `pyproject.toml`:

- `langchain` - Core LangChain framework
- `langgraph` - Graph-based agent workflows
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community integrations (FAISS)
- `chromadb` - Vector database
- `faiss-cpu` - Facebook AI Similarity Search
- `openai` - OpenAI Python SDK
- `python-dotenv` - Environment variable management
- `requests` - HTTP client
- `ddgs` - DuckDuckGo search

---

## License

MIT License
