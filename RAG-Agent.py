"""
OpenAI RAG Agent - Retrieval-Augmented Generation with Markdown Files
Uses OpenAI embeddings and ChromaDB for vector storage
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB settings
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "markdown_rag"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks for better context preservation.

    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Try to break at a paragraph or sentence boundary
        if end < text_length:
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

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_length else text_length

    return chunks


def load_markdown_file(file_path: str) -> str:
    """Load content from a markdown file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.suffix.lower() in ['.md', '.markdown']:
        raise ValueError(f"Expected markdown file, got: {path.suffix}")

    return path.read_text(encoding='utf-8')


class RAGAgent:
    """
    RAG Agent that uses OpenAI embeddings and ChromaDB for retrieval.
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """Initialize the RAG agent with ChromaDB."""
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Create OpenAI embedding function for ChromaDB
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBEDDING_MODEL
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "RAG collection for markdown documents"}
        )

        print(f"Initialized RAG Agent with {self.collection.count()} documents in collection")

    def add_document(self, file_path: str, chunk_size: int = 1000, overlap: int = 200) -> int:
        """
        Add a markdown document to the RAG database.

        Args:
            file_path: Path to the markdown file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks

        Returns:
            Number of chunks added
        """
        # Load and chunk the document
        content = load_markdown_file(file_path)
        chunks = chunk_text(content, chunk_size, overlap)

        if not chunks:
            print("No content to add")
            return 0

        # Generate unique IDs for each chunk
        file_name = Path(file_path).name
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        # Prepare metadata for each chunk
        metadatas = [
            {
                "source": file_path,
                "file_name": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]

        # Add to collection (ChromaDB handles embedding automatically)
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        print(f"Added {len(chunks)} chunks from '{file_name}' to the database")
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Search for relevant documents based on the query.

        Args:
            query: The search query
            n_results: Number of results to return

        Returns:
            List of relevant document chunks with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })

        return formatted_results

    def query(self, question: str, n_context: int = 5) -> str:
        """
        Query the RAG agent with a question.

        Args:
            question: The question to ask
            n_context: Number of context documents to retrieve

        Returns:
            Generated answer based on retrieved context
        """
        # Retrieve relevant context
        context_docs = self.search(question, n_results=n_context)

        if not context_docs:
            return "No relevant information found in the knowledge base."

        # Build context string
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc['metadata'].get('file_name', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{doc['content']}")

        context_str = "\n\n---\n\n".join(context_parts)

        # Generate response using OpenAI
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer the question.
If the context doesn't contain enough information to fully answer the question, say so.
Always be accurate and cite which source(s) you used when relevant."""

        user_prompt = f"""Context:
{context_str}

---

Question: {question}

Please provide a comprehensive answer based on the context above."""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def clear_collection(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        print("Collection cleared")

    def list_sources(self) -> list[str]:
        """List all source files in the collection."""
        if self.collection.count() == 0:
            return []

        # Get all documents with metadata
        all_data = self.collection.get(include=["metadatas"])
        sources = set()
        for metadata in all_data['metadatas']:
            if metadata and 'file_name' in metadata:
                sources.add(metadata['file_name'])

        return list(sources)


def interactive_mode(agent: RAGAgent):
    """Run the agent in interactive mode."""
    print("\n" + "="*50)
    print("RAG Agent Interactive Mode")
    print("="*50)
    print("Commands:")
    print("  /add <file_path>  - Add a markdown file to the knowledge base")
    print("  /search <query>   - Search for relevant documents")
    print("  /sources          - List all sources in the knowledge base")
    print("  /clear            - Clear all documents")
    print("  /quit             - Exit the agent")
    print("  (any other input) - Ask a question")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == '/quit':
                print("Goodbye!")
                break

            elif user_input.startswith('/add '):
                file_path = user_input[5:].strip()
                try:
                    agent.add_document(file_path)
                except Exception as e:
                    print(f"Error adding document: {e}")

            elif user_input.startswith('/search '):
                query = user_input[8:].strip()
                results = agent.search(query)
                print(f"\nFound {len(results)} relevant chunks:")
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} (distance: {result['distance']:.4f}) ---")
                    print(f"Source: {result['metadata'].get('file_name', 'Unknown')}")
                    print(f"Content: {result['content'][:200]}...")

            elif user_input.lower() == '/sources':
                sources = agent.list_sources()
                if sources:
                    print("Sources in knowledge base:")
                    for source in sources:
                        print(f"  - {source}")
                else:
                    print("No sources in knowledge base")

            elif user_input.lower() == '/clear':
                confirm = input("Are you sure you want to clear all documents? (yes/no): ")
                if confirm.lower() == 'yes':
                    agent.clear_collection()

            else:
                # Regular question
                print("\nAgent: ", end="")
                answer = agent.query(user_input)
                print(answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI RAG Agent for Markdown Files")
    parser.add_argument('--add', type=str, help='Add a markdown file to the knowledge base')
    parser.add_argument('--query', type=str, help='Query the knowledge base')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--clear', action='store_true', help='Clear the knowledge base')

    args = parser.parse_args()

    # Initialize agent
    agent = RAGAgent()

    if args.clear:
        agent.clear_collection()
        return

    if args.add:
        agent.add_document(args.add)

    if args.query:
        answer = agent.query(args.query)
        print(f"\nAnswer: {answer}")

    if args.interactive or (not args.add and not args.query and not args.clear):
        interactive_mode(agent)


if __name__ == "__main__":
    main()
