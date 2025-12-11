"""
Multi-LLM RAG Agent
===================
A comprehensive multi-LLM agent system that integrates:
- OpenAI: For chat and RAG (Retrieval-Augmented Generation) using LangGraph
- Google Gemini: For image generation (Nano Banana) and video generation (Veo 3.1)
- Claude: For coding, reasoning, and web search capabilities

Features:
- LangGraph-based stateful agent workflow
- RAG with vector store for document retrieval
- Interactive CLI with file loading commands (/load, /image, /video, etc.)
- Image generation and editing with Gemini Flash
- Video generation with Veo 3.1
- Claude web search for real-time information
- Comprehensive logging for all tool calls

Usage:
    python Multi-LLM-RAG-Agent.py [--no-auto-load] [--load FILE] [--query QUERY]

Author: Multi-LLM Agent System
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Annotated, Literal, Optional
from io import BytesIO
from typing_extensions import TypedDict

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Third-party imports
import requests
from PIL import Image

# LangChain and LangGraph imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph

# Google Gemini imports
from google import genai
from google.genai import types

# Anthropic/Claude imports
import anthropic

# DuckDuckGo for web search fallback
from ddgs import DDGS


# ============================================
# LOGGING CONFIGURATION
# ============================================

def setup_logging() -> logging.Logger:
    """Configure and return the logger with appropriate formatting."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create logger
    logger = logging.getLogger("MultiLLMAgent")
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Clear existing handlers
    logger.handlers = []

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration class for all API keys and model settings."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Google Gemini Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
    VEO_VIDEO_MODEL: str = os.getenv("VEO_VIDEO_MODEL", "veo-3.1-generate-preview")

    # Claude/Anthropic Configuration
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

    # RAG Configuration
    DEFAULT_DOCUMENT: str = os.getenv("DEFAULT_DOCUMENT", "VIVEK-VISHWAKARMA-RESUME.md")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Output Configuration
    IMAGE_OUTPUT_DIR: str = os.getenv("IMAGE_OUTPUT_DIR", "generated_images")
    VIDEO_OUTPUT_DIR: str = os.getenv("VIDEO_OUTPUT_DIR", "generated_videos")

    @classmethod
    def validate(cls) -> bool:
        """Validate that all required API keys are set."""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not cls.CLAUDE_API_KEY:
            missing.append("CLAUDE_API_KEY")

        if missing:
            logger.error(f"Missing required API keys: {', '.join(missing)}")
            return False
        return True


# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State schema for the LangGraph agent."""
    messages: Annotated[list, add_messages]


# ============================================
# GOOGLE GEMINI CLIENTS
# ============================================

def get_gemini_client():
    """Get the Google Gemini client."""
    return genai.Client(api_key=Config.GOOGLE_API_KEY)


# ============================================
# CLAUDE CLIENT
# ============================================

def get_claude_client():
    """Get the Anthropic/Claude client."""
    return anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)


# ============================================
# UTILITY TOOLS
# ============================================

@tool
def get_current_date() -> str:
    """Get the current date.
    Use this tool when the user asks about today's date or the current date.
    """
    logger.info("Tool called: get_current_date")
    today = datetime.now()
    result = f"Today's date: {today.strftime('%A, %B %d, %Y')}"
    logger.info(f"Tool result: {result}")
    return result


@tool
def get_current_time() -> str:
    """Get the current exact time.
    Use this tool when the user asks about the current time.
    """
    logger.info("Tool called: get_current_time")
    current_time = datetime.now()
    result = f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    logger.info(f"Tool result: {result}")
    return result


@tool
def get_india_time() -> str:
    """Get the current exact time in India (IST - Indian Standard Time).
    Use this tool when the user asks about the current time in India.
    """
    logger.info("Tool called: get_india_time")
    ist = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(ist)
    result = f"Current time in India (IST): {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST"
    logger.info(f"Tool result: {result}")
    return result


@tool
def get_weather(city: str = "Delhi", country: str = "India") -> str:
    """Get the current weather in a city.
    Use this tool when the user asks about weather.

    Args:
        city: The city name (default: Delhi)
        country: The country name (default: India)
    """
    logger.info(f"Tool called: get_weather(city={city}, country={country})")
    try:
        url = f"https://wttr.in/{city},{country}?format=%C+%t+%h+%w"
        response = requests.get(url, timeout=10, headers={"User-Agent": "curl/7.68.0"})

        if response.status_code == 200:
            weather_data = response.text.strip()
            result = f"Current weather in {city}, {country}: {weather_data}"
        else:
            result = f"Unable to fetch weather for {city}. Please try again."

        logger.info(f"Tool result: {result}")
        return result
    except Exception as e:
        error_msg = f"Error fetching weather: {str(e)}"
        logger.error(error_msg)
        return error_msg


# ============================================
# WEB SEARCH TOOLS
# ============================================

@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or any topic.
    Use this tool when you need to find up-to-date information from the internet.

    Args:
        query: The search query to look up on the web
    """
    logger.info(f"Tool called: web_search(query={query})")
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

        result = "\n\n".join(formatted_results)
        logger.info(f"Tool result: Found {len(results)} results")
        return result
    except Exception as e:
        error_msg = f"Error searching the web: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def fetch_url(url: str) -> str:
    """Fetch and read content from a URL/link.
    Use this tool to access websites, GitHub repositories, or any web page.

    Args:
        url: The full URL to fetch (e.g., https://github.com/example/repo)
    """
    logger.info(f"Tool called: fetch_url(url={url})")
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
        # Remove scripts and styles
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        content = unescape(content)
        # Clean whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        # Truncate if too long
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"

        result = f"Content from {url}:\n\n{content}"
        logger.info(f"Tool result: Fetched {len(content)} characters")
        return result
    except Exception as e:
        error_msg = f"Error fetching URL: {str(e)}"
        logger.error(error_msg)
        return error_msg


# ============================================
# RAG KNOWLEDGE BASE
# ============================================

class RAGKnowledgeBase:
    """Manages the RAG knowledge base from markdown/text files."""

    def __init__(self):
        logger.info("Initializing RAG Knowledge Base")
        self.embeddings = OpenAIEmbeddings(model=Config.OPENAI_EMBEDDING_MODEL)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.loaded_files: list[str] = []

    def load_document(self, file_path: str) -> int:
        """Load a document (markdown, text, etc.) into the knowledge base.

        Args:
            file_path: Path to the document file

        Returns:
            Number of chunks created
        """
        logger.info(f"Loading document: {file_path}")
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

        self.loaded_files.append(str(path.name))
        logger.info(f"Loaded {len(documents)} chunks from '{path.name}'")
        return len(documents)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Search the knowledge base for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def get_loaded_files(self) -> list[str]:
        """Get list of loaded file names."""
        return self.loaded_files


# Global knowledge base instance
knowledge_base = RAGKnowledgeBase()


@tool
def search_knowledge_base(query: str) -> str:
    """Search the loaded documents for relevant information.
    Use this tool to find information from the loaded knowledge base (resume, documents).

    Args:
        query: The search query to find relevant information
    """
    logger.info(f"Tool called: search_knowledge_base(query={query})")

    if knowledge_base.vector_store is None:
        return "No documents loaded in the knowledge base. Please load a document first using /load command."

    docs = knowledge_base.search(query, k=4)

    if not docs:
        return "No relevant information found in the knowledge base."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        results.append(f"[{i}] From {source}:\n{doc.page_content}")

    result = "\n\n---\n\n".join(results)
    logger.info(f"Tool result: Found {len(docs)} relevant chunks")
    return result


# ============================================
# GOOGLE GEMINI - IMAGE GENERATION
# ============================================

class GeminiImageGenerator:
    """Handles image generation using Google Gemini Flash (Nano Banana)."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = Config.GEMINI_IMAGE_MODEL
        self.output_dir = Path(Config.IMAGE_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        # Store last generated image response for video generation
        self.last_generated_response = None
        self.last_generated_path: Optional[str] = None

    def generate_image(self, prompt: str, output_filename: Optional[str] = None) -> str:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt describing the image to generate
            output_filename: Optional custom filename for the output

        Returns:
            Path to the generated image or error message
        """
        logger.info(f"Gemini Image Generation: prompt='{prompt[:50]}...'")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config={"response_modalities": ["IMAGE"]}
            )

            # Store the response for potential video generation
            self.last_generated_response = response

            # Process response
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    logger.info(f"Gemini response text: {part.text}")
                elif part.inline_data is not None:
                    # Generate filename if not provided
                    if output_filename is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"generated_{timestamp}.png"

                    output_path = self.output_dir / output_filename

                    # Save the image
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(str(output_path))

                    # Store the path for potential video generation
                    self.last_generated_path = str(output_path)

                    result = f"Image generated successfully: {output_path}"
                    logger.info(result)
                    return result

            return "Image generation completed but no image data was returned."

        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def edit_image(self, prompt: str, image_path: str, output_filename: Optional[str] = None) -> str:
        """Edit an existing image based on a text prompt.

        Args:
            prompt: The text prompt describing the edit
            image_path: Path to the source image
            output_filename: Optional custom filename for the output

        Returns:
            Path to the edited image or error message
        """
        logger.info(f"Gemini Image Edit: prompt='{prompt[:50]}...', image={image_path}")

        try:
            # Load the source image
            source_image = Image.open(image_path)

            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, source_image],
            )

            # Process response
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    logger.info(f"Gemini response text: {part.text}")
                elif part.inline_data is not None:
                    # Generate filename if not provided
                    if output_filename is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"edited_{timestamp}.png"

                    output_path = self.output_dir / output_filename

                    # Save the image
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(str(output_path))

                    result = f"Image edited successfully: {output_path}"
                    logger.info(result)
                    return result

            return "Image editing completed but no image data was returned."

        except Exception as e:
            error_msg = f"Error editing image: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_last_image_for_video(self):
        """Get the last generated image object for video generation.

        Returns:
            The image object from the last generation, or None if not available
        """
        if self.last_generated_response is None:
            return None

        try:
            # Try using parts directly first (simpler API per Google docs)
            if hasattr(self.last_generated_response, 'parts') and self.last_generated_response.parts:
                part = self.last_generated_response.parts[0]
                if hasattr(part, 'as_image'):
                    return part.as_image()

            # Fallback to candidates structure
            if hasattr(self.last_generated_response, 'candidates'):
                for part in self.last_generated_response.candidates[0].content.parts:
                    if hasattr(part, 'as_image'):
                        return part.as_image()
                    elif part.inline_data is not None:
                        return part

            logger.error("Could not find image in response")
            return None
        except Exception as e:
            logger.error(f"Error getting image for video: {e}")
            return None

    def has_generated_image(self) -> bool:
        """Check if there's a previously generated image available."""
        return self.last_generated_response is not None


# Global image generator instance
image_generator = GeminiImageGenerator()


# ============================================
# GOOGLE GEMINI - VIDEO GENERATION (VEO 3.1)
# ============================================

class VeoVideoGenerator:
    """Handles video generation using Google Veo 3.1."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = Config.VEO_VIDEO_MODEL
        self.output_dir = Path(Config.VIDEO_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    def _wait_for_operation(self, op, label: str = "video"):
        """Wait for a video generation operation to complete."""
        while not op.done:
            logger.info(f"Waiting for {label} generation...")
            time.sleep(10)
            op = self.client.operations.get(op)

        # Check for errors in the operation
        if hasattr(op, 'error') and op.error:
            logger.error(f"{label} generation failed with error: {op.error}")
            raise Exception(f"Operation failed: {op.error}")

        logger.info(f"{label.capitalize()} generated successfully.")
        return op

    def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 8,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        output_filename: Optional[str] = None
    ) -> str:
        """Generate a video from a text prompt.

        Args:
            prompt: The text prompt describing the video
            duration_seconds: Duration of the video (default: 8)
            aspect_ratio: Aspect ratio (default: 16:9)
            resolution: Video resolution (default: 720p)
            output_filename: Optional custom filename

        Returns:
            Path to the generated video or error message
        """
        logger.info(f"Veo Video Generation: prompt='{prompt[:50]}...'")

        try:
            op = self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    duration_seconds=duration_seconds,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                ),
            )

            op = self._wait_for_operation(op, "video")

            # Get the generated video
            video_clip = op.response.generated_videos[0]

            # Generate filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"generated_{timestamp}.mp4"

            output_path = self.output_dir / output_filename

            # Download and save the video
            self.client.files.download(file=video_clip.video)
            video_clip.video.save(str(output_path))

            result = f"Video generated successfully: {output_path}"
            logger.info(result)
            return result

        except Exception as e:
            error_msg = f"Error generating video: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def generate_video_from_image(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        use_last_generated: bool = False,
        resolution: str = "720p",
        output_filename: Optional[str] = None
    ) -> str:
        """Generate a video from an image using Veo 3.1.

        This method allows creating videos from:
        1. A specific image file path
        2. The last image generated by Nano Banana (Gemini)

        Args:
            prompt: The text prompt describing the video animation
            image_path: Optional path to an image file to use as starting frame
            use_last_generated: If True, uses the last Nano Banana generated image
            resolution: Video resolution (default: 720p)
            output_filename: Optional custom filename

        Returns:
            Path to the generated video or error message
        """
        logger.info(f"Veo Image-to-Video: prompt='{prompt[:50]}...'")

        try:
            image_input = None

            # Determine image source
            if use_last_generated:
                # Use the last generated image from Nano Banana
                if not image_generator.has_generated_image():
                    return "No previously generated image available. Please generate an image first using /image command."

                image_input = image_generator.get_last_image_for_video()
                if image_input is None:
                    return "Failed to retrieve the last generated image for video."

                logger.info("Using last generated Nano Banana image for video")

            elif image_path:
                # Load image from file path
                if not Path(image_path).exists():
                    return f"Image file not found: {image_path}"

                # Upload the image file for video generation
                logger.info(f"Loading image from: {image_path}")
                pil_image = Image.open(image_path)

                # For file-based images, we need to use a different approach
                # Generate the image through Gemini first to get the proper format
                response = self.client.models.generate_content(
                    model=Config.GEMINI_IMAGE_MODEL,
                    contents=[f"Reproduce this image exactly as it is", pil_image],
                    config={"response_modalities": ["IMAGE"]}
                )

                # Get the image from response
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'as_image'):
                        image_input = part.as_image()
                        break
                    elif part.inline_data is not None:
                        image_input = part
                        break

                if image_input is None:
                    return "Failed to process the image for video generation."

            else:
                return "Please provide either an image_path or set use_last_generated=True"

            # Generate video with Veo 3.1 using the image
            logger.info("Starting Veo 3.1 video generation from image...")

            op = self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                image=image_input,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    resolution=resolution,
                ),
            )

            op = self._wait_for_operation(op, "image-to-video")

            # Check if response is valid
            if op.response is None:
                logger.error("Video generation returned no response")
                return "Video generation failed: No response received from API"

            if not hasattr(op.response, 'generated_videos') or op.response.generated_videos is None:
                logger.error(f"No generated_videos in response. Response: {op.response}")
                return "Video generation failed: No videos in response"

            if len(op.response.generated_videos) == 0:
                logger.error("generated_videos list is empty")
                return "Video generation failed: Empty video list returned"

            # Get the generated video
            video_clip = op.response.generated_videos[0]

            if video_clip is None or not hasattr(video_clip, 'video') or video_clip.video is None:
                logger.error(f"Invalid video clip: {video_clip}")
                return "Video generation failed: Invalid video data received"

            # Generate filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"image_to_video_{timestamp}.mp4"

            output_path = self.output_dir / output_filename

            # Download and save the video
            self.client.files.download(file=video_clip.video)
            video_clip.video.save(str(output_path))

            result = f"Video generated from image successfully: {output_path}"
            logger.info(result)
            return result

        except Exception as e:
            error_msg = f"Error generating video from image: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def generate_and_animate(
        self,
        image_prompt: str,
        video_prompt: str,
        resolution: str = "720p",
        output_filename: Optional[str] = None
    ) -> str:
        """Generate an image with Nano Banana and immediately animate it with Veo 3.1.

        This is a convenience method that combines image generation and video creation
        into a single workflow.

        Args:
            image_prompt: The prompt for generating the initial image
            video_prompt: The prompt describing how to animate the image
            resolution: Video resolution (default: 720p)
            output_filename: Optional custom filename

        Returns:
            Path to the generated video or error message
        """
        logger.info(f"Generate and Animate: image='{image_prompt[:30]}...', video='{video_prompt[:30]}...'")

        try:
            # Step 1: Generate the image with Nano Banana
            logger.info("Step 1: Generating image with Nano Banana...")
            image_result = image_generator.generate_image(image_prompt)

            if "Error" in image_result or "failed" in image_result.lower():
                return f"Image generation failed: {image_result}"

            logger.info(f"Image generated: {image_result}")

            # Step 2: Generate video from the image
            logger.info("Step 2: Generating video with Veo 3.1...")
            video_result = self.generate_video_from_image(
                prompt=video_prompt,
                use_last_generated=True,
                resolution=resolution,
                output_filename=output_filename
            )

            return f"Workflow completed!\nImage: {image_result}\nVideo: {video_result}"

        except Exception as e:
            error_msg = f"Error in generate and animate workflow: {str(e)}"
            logger.error(error_msg)
            return error_msg


# Global video generator instance
video_generator = VeoVideoGenerator()


# ============================================
# CLAUDE - REASONING AND WEB SEARCH
# ============================================

class ClaudeAssistant:
    """Handles Claude-based reasoning, coding, and web search."""

    def __init__(self):
        self.client = get_claude_client()
        self.model = Config.CLAUDE_MODEL

    def reason(self, query: str, context: str = "") -> str:
        """Use Claude for complex reasoning and coding tasks.

        Args:
            query: The question or task
            context: Optional additional context

        Returns:
            Claude's response
        """
        logger.info(f"Claude Reasoning: query='{query[:50]}...'")

        try:
            messages = [
                {
                    "role": "user",
                    "content": f"{context}\n\n{query}" if context else query
                }
            ]

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
            )

            result = ""
            for block in response.content:
                if block.type == "text":
                    result += block.text

            logger.info(f"Claude response: {len(result)} characters")
            return result

        except Exception as e:
            error_msg = f"Error with Claude reasoning: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def web_search(self, query: str) -> str:
        """Use Claude's built-in web search tool for real-time information.

        Args:
            query: The search query

        Returns:
            Search results with citations
        """
        logger.info(f"Claude Web Search: query='{query}'")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }]
            )

            # Process response and extract text with citations
            result = ""
            for block in response.content:
                if block.type == "text":
                    result += block.text
                    # Add citations if available
                    if hasattr(block, 'citations') and block.citations:
                        result += "\n\nSources:\n"
                        for citation in block.citations:
                            if hasattr(citation, 'url') and hasattr(citation, 'title'):
                                result += f"- [{citation.title}]({citation.url})\n"

            logger.info(f"Claude web search response: {len(result)} characters")
            return result

        except Exception as e:
            error_msg = f"Error with Claude web search: {str(e)}"
            logger.error(error_msg)
            # Fallback to DuckDuckGo search
            logger.info("Falling back to DuckDuckGo search")
            return web_search.invoke(query)


# Global Claude assistant instance
claude_assistant = ClaudeAssistant()


# ============================================
# LANGGRAPH TOOLS INTEGRATION
# ============================================

@tool
def generate_image(prompt: str) -> str:
    """Generate an image using Google Gemini (Nano Banana).
    Use this tool when the user wants to create/generate an image.

    Args:
        prompt: A detailed description of the image to generate
    """
    logger.info(f"Tool called: generate_image(prompt={prompt[:50]}...)")
    return image_generator.generate_image(prompt)


@tool
def edit_image(prompt: str, image_path: str) -> str:
    """Edit an existing image using Google Gemini.
    Use this tool when the user wants to modify/edit an existing image.

    Args:
        prompt: A description of the edits to make
        image_path: Path to the source image to edit
    """
    logger.info(f"Tool called: edit_image(prompt={prompt[:50]}..., image_path={image_path})")
    return image_generator.edit_image(prompt, image_path)


@tool
def generate_video(prompt: str, duration: int = 8) -> str:
    """Generate a video using Google Veo 3.1.
    Use this tool when the user wants to create/generate a video.

    Args:
        prompt: A detailed description of the video to generate
        duration: Duration in seconds (default: 8, max: 8)
    """
    logger.info(f"Tool called: generate_video(prompt={prompt[:50]}..., duration={duration})")
    return video_generator.generate_video(prompt, duration_seconds=min(duration, 8))


@tool
def ask_claude(query: str) -> str:
    """Ask Claude for help with coding, reasoning, or complex tasks.
    Use this tool for coding questions, debugging, explanations, or complex reasoning.

    Args:
        query: The question or task for Claude
    """
    logger.info(f"Tool called: ask_claude(query={query[:50]}...)")
    return claude_assistant.reason(query)


@tool
def claude_web_search(query: str) -> str:
    """Use Claude's web search for real-time information with citations.
    Use this tool when you need up-to-date information from the web with reliable sources.

    Args:
        query: The search query for real-time information
    """
    logger.info(f"Tool called: claude_web_search(query={query})")
    return claude_assistant.web_search(query)


@tool
def generate_video_from_image(prompt: str, image_path: Optional[str] = None) -> str:
    """Generate a video from an image using Google Veo 3.1.
    Use this tool to animate an existing image or the last generated image.

    If no image_path is provided, it will use the last image generated by Nano Banana.

    Args:
        prompt: A description of how to animate the image (motion, camera movement, etc.)
        image_path: Optional path to an image file. If not provided, uses the last generated image.
    """
    logger.info(f"Tool called: generate_video_from_image(prompt={prompt[:50]}...)")

    if image_path:
        return video_generator.generate_video_from_image(prompt=prompt, image_path=image_path)
    else:
        return video_generator.generate_video_from_image(prompt=prompt, use_last_generated=True)


@tool
def animate_image(image_prompt: str, animation_prompt: str) -> str:
    """Generate an image and immediately animate it into a video.
    Use this tool when the user wants to create an animated video from a concept.

    This combines Nano Banana (image generation) and Veo 3.1 (video generation) in one step.

    Args:
        image_prompt: A description of the image to generate
        animation_prompt: A description of how to animate the image
    """
    logger.info(f"Tool called: animate_image(image='{image_prompt[:30]}...', animation='{animation_prompt[:30]}...')")
    return video_generator.generate_and_animate(image_prompt=image_prompt, video_prompt=animation_prompt)


# ============================================
# LANGGRAPH AGENT SETUP
# ============================================

SYSTEM_PROMPT = """You are a powerful Multi-LLM AI Assistant with access to multiple AI models and tools.

You have access to the following capabilities:

1. **Knowledge Base Search**: Search loaded documents (resumes, notes, etc.) for relevant information.
2. **Date/Time Tools**: Get current date, time, and India time.
3. **Weather**: Get current weather for any city.
4. **Web Search**: Search the internet for current information using DuckDuckGo.
5. **URL Fetch**: Read content from any web URL.
6. **Image Generation**: Create images using Google Gemini (Nano Banana) - use for creative image requests.
7. **Image Editing**: Modify existing images with text prompts.
8. **Video Generation**: Create videos using Google Veo 3.1 - use for video creation requests.
9. **Image-to-Video**: Animate an existing image or the last generated image into a video using Veo 3.1.
10. **Generate & Animate**: Create an image with Nano Banana and immediately animate it with Veo 3.1.
11. **Claude Reasoning**: Ask Claude for coding help, debugging, explanations, or complex reasoning.
12. **Claude Web Search**: Get real-time web information with citations from Claude.

Guidelines:
- For personal/document questions, ALWAYS use search_knowledge_base first.
- For date queries, use get_current_date.
- For time queries, use get_current_time or get_india_time.
- For weather, use get_weather.
- For general web searches, use web_search.
- For creating images, use generate_image.
- For editing images, use edit_image with the image path.
- For creating videos from text, use generate_video.
- For creating videos from images, use generate_video_from_image.
- For generating an image and animating it in one step, use animate_image.
- For coding/reasoning questions, use ask_claude.
- For real-time news with citations, use claude_web_search.

Be helpful, accurate, and friendly. When using tools, explain what you're doing."""


# All available tools
ALL_TOOLS = [
    get_current_date,
    get_current_time,
    get_india_time,
    get_weather,
    search_knowledge_base,
    web_search,
    fetch_url,
    generate_image,
    edit_image,
    generate_video,
    generate_video_from_image,
    animate_image,
    ask_claude,
    claude_web_search,
]


def create_agent_graph() -> CompiledStateGraph:
    """Create the LangGraph agent workflow."""
    logger.info("Creating LangGraph agent workflow")

    # Initialize the LLM with tools
    llm = ChatOpenAI(model=Config.OPENAI_CHAT_MODEL, temperature=0.7)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> AgentState:
        """The agent node that processes messages and decides on actions."""
        messages = state["messages"]

        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM made tool calls, route to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "__end__"

    # Create the tool node
    tool_node = ToolNode(ALL_TOOLS)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    logger.info("LangGraph agent workflow created successfully")
    return graph.compile()


# ============================================
# INTERACTIVE CLI
# ============================================

def print_help():
    """Print help information."""
    print("\n" + "=" * 70)
    print("MULTI-LLM RAG AGENT - COMMANDS")
    print("=" * 70)
    print("""
DOCUMENT COMMANDS:
  /load <file_path>     - Load a document into the knowledge base
  /files                - List all loaded documents

IMAGE COMMANDS:
  /image <prompt>       - Generate an image from a text prompt
  /edit <path> <prompt> - Edit an existing image

VIDEO COMMANDS:
  /video <prompt>       - Generate a video from a text prompt
  /animate              - Animate the last generated image into a video
  /animate <path> <prompt> - Animate a specific image into a video
  /create_animate <image_prompt> | <video_prompt>
                        - Generate an image and animate it in one step

CLAUDE COMMANDS:
  /claude <query>       - Ask Claude directly for reasoning/coding help
  /websearch <query>    - Use Claude's web search for real-time info

UTILITY COMMANDS:
  /help                 - Show this help message
  /clear                - Clear conversation history
  /quit                 - Exit the agent

GENERAL CHAT:
  Just type your message to chat with the agent.
  The agent will automatically use the appropriate tools.
""")
    print("=" * 70 + "\n")


def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "=" * 70)
    print("MULTI-LLM RAG AGENT")
    print("=" * 70)
    print("Welcome! I'm a Multi-LLM AI assistant powered by OpenAI, Google Gemini, and Claude.")
    print("Type /help for available commands or just start chatting!")
    print("=" * 70 + "\n")

    # Validate configuration
    if not Config.validate():
        print("Error: Missing required API keys. Please check your .env file.")
        return

    # Create the agent graph
    agent = create_agent_graph()

    # Conversation history
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            elif user_input.lower() == "/help":
                print_help()
                continue

            elif user_input.lower() == "/clear":
                messages = []
                print("Conversation history cleared.")
                continue

            elif user_input.lower() == "/files":
                files = knowledge_base.get_loaded_files()
                if files:
                    print(f"Loaded files: {', '.join(files)}")
                else:
                    print("No files loaded.")
                continue

            elif user_input.startswith("/load "):
                file_path = user_input[6:].strip()
                try:
                    num_chunks = knowledge_base.load_document(file_path)
                    print(f"Loaded {num_chunks} chunks from '{file_path}'")
                except Exception as e:
                    print(f"Error loading file: {e}")
                continue

            elif user_input.startswith("/image "):
                prompt = user_input[7:].strip()
                print("Generating image...")
                result = image_generator.generate_image(prompt)
                print(f"Result: {result}")
                continue

            elif user_input.startswith("/edit "):
                parts = user_input[6:].strip().split(" ", 1)
                if len(parts) < 2:
                    print("Usage: /edit <image_path> <prompt>")
                    continue
                image_path, prompt = parts
                print("Editing image...")
                result = image_generator.edit_image(prompt, image_path)
                print(f"Result: {result}")
                continue

            elif user_input.startswith("/video "):
                prompt = user_input[7:].strip()
                print("Generating video (this may take a few minutes)...")
                result = video_generator.generate_video(prompt)
                print(f"Result: {result}")
                continue

            elif user_input.startswith("/animate "):
                # /animate <path> <prompt> - Animate a specific image
                parts = user_input[9:].strip().split(" ", 1)
                if len(parts) < 2:
                    print("Usage: /animate <image_path> <prompt>")
                    print("       Or use just '/animate <prompt>' to animate the last generated image")
                    continue
                image_path, prompt = parts
                print("Generating video from image (this may take a few minutes)...")
                result = video_generator.generate_video_from_image(prompt=prompt, image_path=image_path)
                print(f"Result: {result}")
                continue

            elif user_input == "/animate" or (user_input.startswith("/animate") and " " not in user_input[9:].strip()):
                # /animate or /animate <prompt> - Animate the last generated image
                prompt = user_input[8:].strip() if len(user_input) > 8 else ""
                if not prompt:
                    prompt = input("Enter animation prompt: ").strip()
                if not prompt:
                    print("Animation prompt is required.")
                    continue
                if not image_generator.has_generated_image():
                    print("No previously generated image. Use /image first or provide an image path.")
                    continue
                print("Generating video from last generated image (this may take a few minutes)...")
                result = video_generator.generate_video_from_image(prompt=prompt, use_last_generated=True)
                print(f"Result: {result}")
                continue

            elif user_input.startswith("/create_animate "):
                # /create_animate <image_prompt> | <video_prompt>
                content = user_input[16:].strip()
                if "|" not in content:
                    print("Usage: /create_animate <image_prompt> | <video_prompt>")
                    print("Example: /create_animate a cute cat sitting | the cat slowly turns its head")
                    continue
                parts = content.split("|", 1)
                image_prompt = parts[0].strip()
                video_prompt = parts[1].strip()
                if not image_prompt or not video_prompt:
                    print("Both image prompt and video prompt are required.")
                    continue
                print("Generating image and animating it (this may take several minutes)...")
                result = video_generator.generate_and_animate(image_prompt=image_prompt, video_prompt=video_prompt)
                print(f"Result: {result}")
                continue

            elif user_input.startswith("/claude "):
                query = user_input[8:].strip()
                print("Asking Claude...")
                result = claude_assistant.reason(query)
                print(f"\nClaude: {result}")
                continue

            elif user_input.startswith("/websearch "):
                query = user_input[11:].strip()
                print("Searching with Claude...")
                result = claude_assistant.web_search(query)
                print(f"\nClaude: {result}")
                continue

            # Regular chat - use the agent
            messages.append(HumanMessage(content=user_input))

            # Invoke the agent
            result = agent.invoke({"messages": messages})

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
                messages = final_messages

            # Keep history manageable (last 20 messages)
            if len(messages) > 20:
                if isinstance(messages[0], SystemMessage):
                    messages = [messages[0]] + messages[-19:]
                else:
                    messages = messages[-20:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


def auto_load_default_document():
    """Auto-load the default document on startup."""
    script_dir = Path(__file__).parent
    default_doc = script_dir / Config.DEFAULT_DOCUMENT

    if default_doc.exists():
        try:
            num_chunks = knowledge_base.load_document(str(default_doc))
            print(f"Auto-loaded {num_chunks} chunks from '{Config.DEFAULT_DOCUMENT}'")
            return True
        except Exception as e:
            logger.warning(f"Could not auto-load default document: {e}")
            return False
    else:
        logger.info(f"Default document not found: {default_doc}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-LLM RAG Agent with OpenAI, Google Gemini, and Claude"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load a document file at startup"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query mode - execute a query and exit"
    )
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help="Skip auto-loading the default document"
    )

    args = parser.parse_args()

    logger.info("Starting Multi-LLM RAG Agent")

    # Validate configuration
    if not Config.validate():
        print("Error: Missing required API keys. Please check your .env file.")
        sys.exit(1)

    # Auto-load default document unless disabled
    if not args.no_auto_load:
        auto_load_default_document()

    # Load additional file if specified
    if args.load:
        try:
            num_chunks = knowledge_base.load_document(args.load)
            print(f"Loaded {num_chunks} chunks from '{args.load}'")
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    # Single query mode
    if args.query:
        agent = create_agent_graph()
        result = agent.invoke({"messages": [HumanMessage(content=args.query)]})

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                print(f"\nAgent: {msg.content}")
                break
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
