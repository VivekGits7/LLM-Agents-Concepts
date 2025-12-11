# Multi-LLM RAG Agent - Model & Tool Flow

## Overview of  Multi-LLM RAG Agent

A comprehensive multi-LLM agent system using **LangGraph** for stateful agent workflow orchestration.

---

## Models Used

| Provider | Model | Purpose |
|----------|-------|---------|
| **OpenAI** | `gpt-4o-mini` | Main chat orchestrator & RAG |
| **OpenAI** | `text-embedding-3-small` | Document embeddings for RAG |
| **Google Gemini** | `gemini-2.5-flash-image` | Image generation (Nano Banana) |
| **Google Veo** | `veo-3.1-generate-preview` | Video generation |
| **Claude** | `claude-sonnet-4-5-20250929` | Reasoning, coding & web search |

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE CLI                              │
│  Commands: /load, /image, /video, /animate, /claude, /websearch │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│   DIRECT COMMAND        │     │      LANGGRAPH AGENT            │
│   (Bypass Agent)        │     │      (Chat Messages)            │
│                         │     │                                 │
│  • /image → Gemini      │     │  ┌───────────────────────────┐  │
│  • /video → Veo         │     │  │     AGENT NODE            │  │
│  • /claude → Claude     │     │  │   (OpenAI gpt-4o-mini)    │  │
│  • /websearch → Claude  │     │  │   + System Prompt         │  │
│  • /load → RAG KB       │     │  │   + Tool Bindings         │  │
│                         │     │  └───────────┬───────────────┘  │
└─────────────────────────┘     │              │                  │
                                │              ▼                  │
                                │  ┌───────────────────────────┐  │
                                │  │   CONDITIONAL ROUTER      │  │
                                │  │   (should_continue)       │  │
                                │  └───────────┬───────────────┘  │
                                │              │                  │
                                │     ┌────────┴────────┐         │
                                │     │                 │         │
                                │     ▼                 ▼         │
                                │  ┌──────┐      ┌───────────┐    │
                                │  │ END  │      │ TOOL NODE │    │
                                │  └──────┘      └─────┬─────┘    │
                                │                      │          │
                                │                      ▼          │
                                │              (Execute Tool)     │
                                │                      │          │
                                │                      └──────────┼──► Back to AGENT NODE
                                └─────────────────────────────────┘
```

---

## LangGraph State Machine

```
         START
           │
           ▼
    ┌──────────────┐
    │    AGENT     │◄────────────────┐
    │              │                 │
    └──────┬───────┘                 │
           │                         │
           ▼                         │
    ┌──────────────┐                 │
    │  has_tools?  │                 │
    └──────┬───────┘                 │
           │                         │
     ┌─────┴─────┐                   │
     │           │                   │
    Yes          No                  │
     │           │                   │
     ▼           ▼                   │
┌─────────┐   ┌─────┐                │
│  TOOLS  │   │ END │                │
└────┬────┘   └─────┘                │
     │                               │
     └───────────────────────────────┘
```

---

## Tools Inventory

### Utility Tools
| Tool | Function | Provider |
|------|----------|----------|
| `get_current_date` | Returns today's date | Local |
| `get_current_time` | Returns current time | Local |
| `get_india_time` | Returns IST time | Local |
| `get_weather` | Fetches weather via wttr.in | HTTP API |

### RAG Tools
| Tool | Function | Provider |
|------|----------|----------|
| `search_knowledge_base` | Semantic search in loaded documents | OpenAI Embeddings + FAISS |

### Web Tools
| Tool | Function | Provider |
|------|----------|----------|
| `web_search` | DuckDuckGo web search | DuckDuckGo |
| `fetch_url` | Fetch & parse URL content | HTTP |
| `claude_web_search` | Web search with citations | Claude API |

### Image Tools
| Tool | Function | Provider |
|------|----------|----------|
| `generate_image` | Text-to-image generation | Gemini Flash (Nano Banana) |
| `edit_image` | Image editing with prompts | Gemini Flash |

### Video Tools
| Tool | Function | Provider |
|------|----------|----------|
| `generate_video` | Text-to-video generation | Veo 3.1 |
| `generate_video_from_image` | Image-to-video animation | Veo 3.1 |
| `animate_image` | Generate image + animate (combined) | Gemini + Veo 3.1 |

### AI Assistant Tools
| Tool | Function | Provider |
|------|----------|----------|
| `ask_claude` | Reasoning, coding, complex tasks | Claude Sonnet |

---

## Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Documents  │    │    Web      │    │   Weather   │    │    User     │  │
│  │  (MD, TXT)  │    │  (URLs)     │    │   (wttr.in) │    │   Prompts   │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │         │
└─────────┼──────────────────┼──────────────────┼──────────────────┼─────────┘
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ RAG Pipeline    │    │ HTTP Requests   │    │ LangGraph Agent         │  │
│  │                 │    │                 │    │                         │  │
│  │ • Text Splitter │    │ • DuckDuckGo    │    │ • Message State         │  │
│  │ • Embeddings    │    │ • URL Fetch     │    │ • Tool Selection        │  │
│  │ • FAISS Store   │    │ • Weather API   │    │ • Response Generation   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AI MODELS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │    OpenAI       │  │  Google Gemini  │  │         Claude              │  │
│  │                 │  │                 │  │                             │  │
│  │ • Chat (GPT-4o) │  │ • Image Gen     │  │ • Reasoning                 │  │
│  │ • Embeddings    │  │ • Image Edit    │  │ • Coding                    │  │
│  │                 │  │ • Video Gen     │  │ • Web Search                │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUTS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Text Response  │  │ Generated Images│  │    Generated Videos         │  │
│  │                 │  │                 │  │                             │  │
│  │  (CLI Output)   │  │ ./generated_    │  │ ./generated_videos/*.mp4    │  │
│  │                 │  │   images/*.png  │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## RAG Knowledge Base Flow

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Document   │────►│  Text Splitter   │────►│    Chunks       │
│  (MD/TXT)    │     │  (1000 chars,    │     │  (Documents)    │
│              │     │   200 overlap)   │     │                 │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │    OpenAI       │
                                              │   Embeddings    │
                                              │ (text-embed-3)  │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  FAISS Vector   │
                                              │     Store       │
                                              └────────┬────────┘
                                                       │
                     ┌─────────────────────────────────┘
                     │
                     ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Query     │────►│ Similarity Search│────►│  Top K Results  │
│              │     │    (k=4)         │     │                 │
└──────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Image-to-Video Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANIMATE_IMAGE WORKFLOW                       │
└─────────────────────────────────────────────────────────────────┘

Step 1: Image Generation (Nano Banana)
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Image Prompt │────►│  Gemini Flash    │────►│  Generated PNG  │
│              │     │  (Nano Banana)   │     │  + Response Obj │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       │ (stored in memory)
                                                       ▼
Step 2: Video Generation (Veo 3.1)
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Video Prompt │────►│    Veo 3.1       │────►│  Generated MP4  │
│ + Image Obj  │     │   (async op)     │     │                 │
└──────────────┘     └──────────────────┘     └─────────────────┘
```

---

## CLI Commands Reference

| Command | Description | Backend |
|---------|-------------|---------|
| `/load <file>` | Load document to RAG | RAGKnowledgeBase |
| `/files` | List loaded documents | RAGKnowledgeBase |
| `/image <prompt>` | Generate image | GeminiImageGenerator |
| `/edit <path> <prompt>` | Edit image | GeminiImageGenerator |
| `/video <prompt>` | Generate video | VeoVideoGenerator |
| `/animate` | Animate last image | VeoVideoGenerator |
| `/animate <path> <prompt>` | Animate specific image | VeoVideoGenerator |
| `/create_animate <img>\|<vid>` | Generate + animate | VeoVideoGenerator |
| `/claude <query>` | Direct Claude query | ClaudeAssistant |
| `/websearch <query>` | Claude web search | ClaudeAssistant |
| `/help` | Show help | CLI |
| `/clear` | Clear history | CLI |
| `/quit` | Exit | CLI |

---

## Key Classes

| Class | Responsibility |
|-------|----------------|
| `Config` | API keys & model configuration |
| `AgentState` | LangGraph state schema (messages) |
| `RAGKnowledgeBase` | Document loading, embedding, search |
| `GeminiImageGenerator` | Image generation & editing |
| `VeoVideoGenerator` | Video generation (text & image-to-video) |
| `ClaudeAssistant` | Reasoning & web search |

---

## Dependencies

```
langchain-core, langchain-openai, langchain-community
langgraph
google-genai
anthropic
faiss-cpu
pillow
duckduckgo-search (ddgs)
python-dotenv
requests
```
