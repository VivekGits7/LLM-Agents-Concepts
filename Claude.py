"""
PDF to Text Extractor using Claude API

This module extracts text content from PDF documents using Anthropic's Claude API.
It splits large PDFs into smaller chunks and processes them in parallel for optimal
performance, then concatenates the results in the correct order.

Architecture:
    1. Split PDF into memory chunks (no temp files created)
    2. Upload each chunk to Claude's Files API in parallel
    3. Process each chunk with Claude Haiku 3.5 for text extraction
    4. Concatenate results in page order
    5. Save combined text to output file

Features:
    - In-memory PDF splitting (no temporary files)
    - Parallel processing with configurable worker threads
    - Thread-safe logging for progress tracking
    - Detailed timing and cost breakdown
    - Token usage tracking and cost estimation
    - Configurable via environment variables

Requirements:
    - PyPDF2: PDF reading and splitting
    - anthropic: Claude API client
    - python-dotenv: Environment variable loading

Configuration:
    Edit the variables at the top of this file:
    PDF_PATH          : Path to the input PDF file
    PAGES_PER_FILE    : Number of pages per chunk (default: 5)
    MAX_WORKERS       : Maximum parallel threads (default: 5)
    OUTPUT_TEXT_FILE  : Output filename (default: extracted_text.txt)
    CLAUDE_PROMPT     : Prompt for text extraction

Environment Variables (.env):
    CLAUDE_API_KEY    : Anthropic API key (required)

Usage:
    1. Set PDF_PATH at the top of this file
    2. Add CLAUDE_API_KEY to your .env file
    3. Run the script:

    $ python Claude.py

Output:
    - extracted_text.txt (or custom OUTPUT_TEXT_FILE)
    - Console output with progress, timing, and cost breakdown

Cost Model:
    Uses Claude Haiku 3.5 pricing:
    - Input:  $0.80 per million tokens
    - Output: $4.00 per million tokens

Functions:
    safe_print(message)
        Thread-safe print function using a lock.

    split_pdf_in_memory(input_path, pages_per_file)
        Split PDF into in-memory byte chunks.

    process_pdf_chunk(chunk_data, api_key, prompt)
        Upload and process a single PDF chunk with Claude API.

    process_pdf_parallel(input_path, pages_per_file, api_key, prompt, max_workers)
        Orchestrate parallel processing of all PDF chunks.

    main()
        Entry point - loads config and runs the extraction pipeline.

Author: [Your Name]
Date: 2025
"""

import sys
import os
import io
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 is not installed.")
    print("Please install it using: pip install PyPDF2")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv is not installed.")
    print("Install it using: pip install python-dotenv")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: anthropic is not installed.")
    print("Please install it using: pip install anthropic")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# ============================================
# CONFIGURATION - Modify these values
# ============================================
PDF_PATH = "your-pdf-file.pdf"  # Path to the input PDF file
PAGES_PER_FILE = 5              # Number of pages per chunk
MAX_WORKERS = 5                 # Maximum parallel threads
OUTPUT_TEXT_FILE = "extracted_text.txt"  # Output filename

# Prompt for Claude to extract text from PDF
CLAUDE_PROMPT = "Extract all text content from this PDF document. Return only the text content without any additional commentary or formatting."

# 🔑 API Key from .env file
CLAUDE_API_KEY: str | None = os.getenv("CLAUDE_API_KEY")
# ============================================

# Thread-safe print lock
print_lock = Lock()


def safe_print(message):
    """Thread-safe print function"""
    with print_lock:
        print(message)


def split_pdf_in_memory(input_path, pages_per_file):
    """
    Split a PDF into multiple in-memory PDF chunks.

    Args:
        input_path (str): Path to the input PDF file
        pages_per_file (int): Number of pages in each chunk

    Returns:
        list: List of tuples (chunk_num, pdf_bytes, start_page, end_page)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File '{input_path}' not found.")

    if not input_path.suffix.lower() == '.pdf':
        raise ValueError("Input file must be a PDF.")

    # Read the input PDF
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)

    # Calculate number of chunks
    num_chunks = (total_pages + pages_per_file - 1) // pages_per_file

    chunks = []

    # Split the PDF into memory chunks
    for chunk_num in range(num_chunks):
        writer = PdfWriter()

        # Calculate page range for this chunk
        start_page = chunk_num * pages_per_file
        end_page = min(start_page + pages_per_file, total_pages)

        # Add pages to the writer
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # Write to in-memory bytes buffer
        pdf_buffer = io.BytesIO()
        writer.write(pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()

        chunks.append((chunk_num + 1, pdf_bytes, start_page + 1, end_page))

    return chunks, total_pages, input_path.name


def process_pdf_chunk(chunk_data, api_key, prompt):
    """
    Upload and process a single PDF chunk with Claude API.

    Args:
        chunk_data: Tuple of (chunk_num, pdf_bytes, start_page, end_page)
        api_key: Claude API key
        prompt: Prompt to send to Claude for text extraction

    Returns:
        tuple: (chunk_num, extracted_text, processing_time, input_tokens, output_tokens)
    """
    chunk_num, pdf_bytes, start_page, end_page = chunk_data

    start_time = time.time()

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Upload the PDF chunk
        safe_print(f"[Part {chunk_num}] Uploading PDF chunk (pages {start_page}-{end_page})...")

        pdf_file = io.BytesIO(pdf_bytes)
        file_response = client.beta.files.upload(
            file=(f"chunk_{chunk_num}.pdf", pdf_file, "application/pdf"),
        )
        file_id = file_response.id

        safe_print(f"[Part {chunk_num}] Uploaded successfully. File ID: {file_id}")

        # Process with Claude
        safe_print(f"[Part {chunk_num}] Extracting text with Claude API...")

        response = client.beta.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "file",
                                "file_id": file_id
                            }
                        }
                    ]
                }
            ],
            betas=["files-api-2025-04-14"],
        )

        # Extract text from response
        extracted_text = ""
        for block in response.content:
            if block.type == "text":
                extracted_text += block.text

        # Get token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        elapsed_time = time.time() - start_time
        safe_print(f"[Part {chunk_num}] Completed in {elapsed_time:.2f}s | Tokens: {input_tokens} in, {output_tokens} out")

        return (chunk_num, extracted_text, elapsed_time, input_tokens, output_tokens)

    except Exception as e:
        elapsed_time = time.time() - start_time
        safe_print(f"[Part {chunk_num}] Error: {e}")
        return (chunk_num, f"[ERROR processing chunk {chunk_num}: {e}]", elapsed_time, 0, 0)


def process_pdf_parallel(input_path, pages_per_file, api_key, prompt, max_workers=5):
    """
    Process entire PDF by splitting and processing chunks in parallel.

    Args:
        input_path: Path to input PDF
        pages_per_file: Pages per chunk
        api_key: Claude API key
        prompt: Prompt to send to Claude
        max_workers: Maximum parallel workers

    Returns:
        str: Concatenated text from all chunks
    """
    total_start_time = time.time()

    # Step 1: Split PDF in memory
    print("\n" + "="*60)
    print("STEP 1: Splitting PDF in memory")
    print("="*60)
    split_start = time.time()

    chunks, total_pages, filename = split_pdf_in_memory(input_path, pages_per_file)

    split_time = time.time() - split_start
    print(f"PDF: {filename}")
    print(f"Total pages: {total_pages}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Pages per chunk: {pages_per_file}")
    print(f"Split time: {split_time:.2f}s\n")

    # Step 2: Process chunks in parallel
    print("="*60)
    print("STEP 2: Processing chunks with Claude API (in parallel)")
    print("="*60)
    process_start = time.time()

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_pdf_chunk, chunk, api_key, prompt): chunk[0]
            for chunk in chunks
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            result = future.result()
            results.append(result)

    process_time = time.time() - process_start
    print(f"\nAll chunks processed in {process_time:.2f}s\n")

    # Step 3: Sort and concatenate results
    print("="*60)
    print("STEP 3: Concatenating results in order")
    print("="*60)
    concat_start = time.time()

    # Sort by chunk number to maintain order
    results.sort(key=lambda x: x[0])

    # Concatenate all text and calculate total tokens
    combined_text = ""
    total_input_tokens = 0
    total_output_tokens = 0

    for chunk_num, text, chunk_time, input_tokens, output_tokens in results:
        combined_text += f"\n--- Part {chunk_num} ---\n"
        combined_text += text
        combined_text += "\n"
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    total_tokens = total_input_tokens + total_output_tokens

    concat_time = time.time() - concat_start
    print(f"Concatenation time: {concat_time:.2f}s\n")

    # Calculate total time
    total_time = time.time() - total_start_time

    # Calculate costs (Claude Haiku 3.5 pricing as of 2024)
    # Input: $0.80 per million tokens
    # Output: $4.00 per million tokens
    input_cost = (total_input_tokens / 1_000_000) * 0.80
    output_cost = (total_output_tokens / 1_000_000) * 4.00
    total_cost = input_cost + output_cost

    # Step 4: Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Split time:         {split_time:.2f}s")
    print(f"Processing time:    {process_time:.2f}s")
    print(f"Concatenation time: {concat_time:.2f}s")
    print(f"Total time:         {total_time:.2f}s")
    print(f"\nExtracted text length: {len(combined_text)} characters")
    print(f"\nToken Usage:")
    print(f"  Input tokens:     {total_input_tokens:,}")
    print(f"  Output tokens:    {total_output_tokens:,}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"\nCost Breakdown (Claude Haiku 3.5):")
    print(f"  Input cost:       ${input_cost:.4f} ({total_input_tokens:,} tokens @ $0.80/M)")
    print(f"  Output cost:      ${output_cost:.4f} ({total_output_tokens:,} tokens @ $4.00/M)")
    print(f"  Total cost:       ${total_cost:.4f}")
    print()

    return combined_text


def main():
    script_dir = Path(__file__).parent

    # Validate configuration
    if not PDF_PATH or PDF_PATH == "your-pdf-file.pdf":
        print("Error: Please set PDF_PATH at the top of this file")
        sys.exit(1)

    if not CLAUDE_API_KEY:
        print("Error: CLAUDE_API_KEY not set in .env file")
        print("Please add your Claude API key to the .env file")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  PDF_PATH: {PDF_PATH}")
    print(f"  PAGES_PER_FILE: {PAGES_PER_FILE}")
    print(f"  CLAUDE_PROMPT: {CLAUDE_PROMPT}")
    print(f"  MAX_WORKERS: {MAX_WORKERS}")
    print(f"  OUTPUT_FILE: {OUTPUT_TEXT_FILE}")

    try:
        # Process PDF
        combined_text = process_pdf_parallel(input_path=PDF_PATH,
                                             pages_per_file=PAGES_PER_FILE,
                                             api_key=CLAUDE_API_KEY,
                                             prompt=CLAUDE_PROMPT,
                                             max_workers=MAX_WORKERS)

        # Step 5: Save to file
        print("="*60)
        print("STEP 4: Saving to file")
        print("="*60)
        output_path = Path(script_dir) / OUTPUT_TEXT_FILE
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

        print(f"Text saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
