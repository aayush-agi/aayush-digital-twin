"""
CV Digital Twin MCP Server
Takes a CV PDF as input and behaves like a digital twin, answering questions about the person.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import json

from fastmcp import FastMCP

# Load .env file if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# Try importing OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

# Try importing PDF libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Create server
mcp = FastMCP("CV Digital Twin Server")

# Global storage for CV content
cv_content: Optional[str] = None
cv_metadata: dict = {}

# OpenAI configuration
openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global openai_client
    
    if not HAS_OPENAI:
        raise ImportError(
            "OpenAI library not installed. Please install it:\n"
            "  pip install openai"
        )
    
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )
        openai_client = OpenAI(api_key=api_key)
    
    return openai_client


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    pages_text.append(page.extract_text() or "")
                text = "\n\n".join(pages_text)
                return text
        except Exception as e:
            print(f"Error with pdfplumber: {e}", file=sys.stderr)
    
    if HAS_PYPDF2:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_text = []
                for page in pdf_reader.pages:
                    pages_text.append(page.extract_text())
                text = "\n\n".join(pages_text)
                return text
        except Exception as e:
            print(f"Error with PyPDF2: {e}", file=sys.stderr)
    
    raise ImportError(
        "No PDF library available. Please install pdfplumber or PyPDF2:\n"
        "  pip install pdfplumber\n"
        "  or\n"
        "  pip install PyPDF2"
    )


def find_all_pdfs_in_docs() -> list[str]:
    """Find all PDF files in docs directory."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    docs_dir = script_dir / "docs"
    
    if not docs_dir.exists():
        return []
    
    # Find all PDF files in docs directory
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    return [str(pdf_path) for pdf_path in pdf_files]


def load_cv(pdf_path: str) -> None:
    """Load and parse CV from PDF file."""
    global cv_content, cv_metadata
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"CV file not found: {pdf_path}")
    
    print(f"Loading CV from: {pdf_path}", file=sys.stderr)
    cv_content = extract_text_from_pdf(pdf_path)
    
    cv_metadata = {
        "file_path": pdf_path,
        "file_name": os.path.basename(pdf_path),
        "content_length": len(cv_content),
        "loaded": True
    }


def load_all_pdfs_from_docs() -> None:
    """Load and combine all PDFs from docs directory."""
    global cv_content, cv_metadata
    
    pdf_files = find_all_pdfs_in_docs()
    
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in docs/ directory")
    
    print(f"Found {len(pdf_files)} PDF file(s) in docs directory", file=sys.stderr)
    
    all_content_parts = []
    loaded_files = []
    
    for pdf_path in pdf_files:
        try:
            print(f"Loading: {os.path.basename(pdf_path)}", file=sys.stderr)
            content = extract_text_from_pdf(pdf_path)
            all_content_parts.append(f"\n\n--- Content from {os.path.basename(pdf_path)} ---\n\n{content}")
            loaded_files.append(pdf_path)
        except Exception as e:
            print(f"Warning: Failed to load {pdf_path}: {e}", file=sys.stderr)
            continue
    
    if not all_content_parts:
        raise Exception("Failed to load any PDF files from docs directory")
    
    cv_content = "\n".join(all_content_parts)
    
    cv_metadata = {
        "file_paths": loaded_files,
        "file_names": [os.path.basename(f) for f in loaded_files],
        "content_length": len(cv_content),
        "num_files": len(loaded_files),
        "loaded": True
    }
    
    print(f"Successfully loaded {len(loaded_files)} PDF file(s), total content length: {len(cv_content)} characters", file=sys.stderr)


def _chat_with_me_impl(message: str, cv_path: Optional[str] = None) -> str:
    """
    Chat with the digital twin based on your CV.
    If a CV path is provided and no CV is loaded, it will load the CV first.
    Then answers your question or message using OpenAI based on the CV content.
    
    Args:
        message: Your message or question (e.g., "What is your work experience?", 
                 "Tell me about yourself", "What are your skills?")
        cv_path: Optional path to CV PDF file. If not provided, will automatically 
                 scan and load all PDF files in the docs/ directory.
        
    Returns:
        Response from the digital twin based on CV content
    """
    global cv_content, cv_metadata
    
    # Load CV if path provided and CV not already loaded
    if cv_path:
        try:
            load_cv(cv_path)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to load CV: {str(e)}"
            }, indent=2)
    
    # If CV not loaded, try to find and load all PDFs in docs directory
    if cv_content is None:
        try:
            load_all_pdfs_from_docs()
        except FileNotFoundError as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to load PDFs from docs directory: {str(e)}"
            }, indent=2)
    
    try:
        client = get_openai_client()
        
        # Prepare the system prompt
        system_prompt = """You are a digital twin of a person based on their CV/resume. 
Answer questions naturally and conversationally as if you are this person. Don't be too professional, act like a normal person with fun personality."""
        
        # Truncate CV content if too long (to fit within token limits)
        max_cv_length = 12000
        cv_text = cv_content[:max_cv_length] if len(cv_content) > max_cv_length else cv_content
        if len(cv_content) > max_cv_length:
            cv_text += "\n\n[Note: CV content truncated for length]"
        
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"My CV/Resume:\n\n{cv_text}\n\n\nMessage: {message}"}
            ],
            temperature=1.0
        )
        
        answer = response.choices[0].message.content
        
        return json.dumps({
            "message": message,
            "response": answer,
            "source": "CV Digital Twin (OpenAI)",
            "model": response.model
        }, indent=2)
        
    except ImportError as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2)
    except ValueError as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error calling OpenAI API: {str(e)}"
        }, indent=2)


# Google Calendar appointment scheduling link
CALENDAR_BOOKING_URL = "https://calendar.app.google/PYGKGLzp8GDNBeS78"


@mcp.tool
def chat_with_me(message: str, cv_path: Optional[str] = None) -> str:
    """Chat with Aayush Srivastava's digital twin based on their CV."""
    return _chat_with_me_impl(message, cv_path)


@mcp.tool
def schedule_meeting() -> str:
    """
    Schedule a meeting with Aayush Srivastava.
    Returns the Google Calendar booking link to set up a meeting.
    Use this when someone wants to connect, schedule a call, book time, 
    or have a meeting with Aayush.
    
    Returns:
        JSON with the calendar booking URL and instructions
    """
    return json.dumps({
        "status": "success",
        "message": "Here's how to schedule a meeting with Aayush Srivastava:",
        "booking_url": CALENDAR_BOOKING_URL,
        "instructions": [
            "Click the booking link below to access Aayush's calendar",
            "Select a date and time that works for you",
            "Fill in your details and any meeting agenda",
            "You'll receive a calendar invite confirmation"
        ],
        "note": "Looking forward to connecting with you!"
    }, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            load_cv(pdf_path)
            print("CV loaded successfully!", file=sys.stderr)
            print(f"CV file: {pdf_path}", file=sys.stderr)
            print(f"Content length: {len(cv_content)} characters", file=sys.stderr)
        except Exception as e:
            print(f"Error loading CV: {e}", file=sys.stderr)
    else:
        # Try to auto-load all PDFs from docs directory
        try:
            load_all_pdfs_from_docs()
            print("PDFs loaded successfully from docs directory!", file=sys.stderr)
            if cv_metadata.get("file_names"):
                print(f"Loaded files: {', '.join(cv_metadata['file_names'])}", file=sys.stderr)
            print(f"Total content length: {len(cv_content)} characters", file=sys.stderr)
        except Exception as e:
            print(f"Error loading PDFs: {e}", file=sys.stderr)
            print("\nCV Digital Twin MCP Server", file=sys.stderr)
            print("Usage: python main.py <path_to_cv.pdf>", file=sys.stderr)
            print("\nOr place PDF file(s) in the docs/ directory and use as MCP server:", file=sys.stderr)
            print("  - chat_with_me(message): Chat with your digital twin", file=sys.stderr)
            print("  - All PDFs in docs/ will be automatically scanned and loaded", file=sys.stderr)
            print("\nTo run as MCP server:", file=sys.stderr)
            print("  python -m fastmcp run main.py", file=sys.stderr)
            print("  or", file=sys.stderr)
            print("  fastmcp run main.py", file=sys.stderr)
