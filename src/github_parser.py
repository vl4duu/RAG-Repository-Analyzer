from github import Github

from dotenv import load_dotenv
import os

import re
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_text_splitters import Language
except ImportError:
    # Fallback for older versions
    from langchain.text_splitter import Language

load_dotenv()
g = Github(os.getenv("GITHUB_API_KEY"))

# Initialize tiktoken encoder for token counting
ENCODING = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4 and text-embedding-ada-002


def get_repo_files(repo_path):
    """
    Parse the repository and extract content from the specified file types.

    Args:
        repo_path: GitHub repository path in the format username/repo_name.
    Returns:
            data List of dictionaries with textual files' data.
    """
    data = []
    repo = g.get_repo(repo_path)

    def traverse_folder(folder=""):
        """
        Recursively traverses a folder, processing all matching files.
        """
        try:
            contents = repo.get_contents(folder)
            for content in contents:
                # Traverse subdirectories
                if content.type == "dir":
                    traverse_folder(content.path)
                # Process files based on type and extension
                elif content.type == "file":
                    process_file(content)
        except Exception as e:
            print(f"Error accessing folder '{folder}': {e}")

    def process_file(content):
        """
        Processes a single file, decoding its content and categorizing it.
        """
        try:
            # Decode and aggregate file data
            file_data = {
                "file_name": content.path,
                "content": content.decoded_content.decode("utf-8", errors="ignore"),
            }

            data.append(file_data)
        except Exception as e:
            print(f"Error processing file {getattr(content, 'path', 'unknown')}: {e}")

    # Start traversal
    traverse_folder()
    return data


def analyze_repository_volume(repo_files):
    """
    Analyzes repository volume to determine optimal chunking strategy.
    
    Args:
        repo_files: List of repository files with their contents
        
    Returns:
        Dictionary with volume metrics and recommended chunk sizes
    """
    total_files = len(repo_files)
    textual_files = 0
    code_files = 0
    total_tokens = 0
    
    for file in repo_files:
        file_name = file["file_name"]
        content = file["content"]
        
        # Count tokens
        tokens = ENCODING.encode(content)
        total_tokens += len(tokens)
        
        # Categorize files
        if file_name.endswith((".md", ".txt", ".xml", ".rst", ".adoc")):
            textual_files += 1
        elif file_name.endswith((".py", ".js", ".java", ".html", ".css", ".ts", ".tsx", 
                                  ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".rb", ".php",
                                  ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh")):
            code_files += 1
    
    # Determine volume category and chunk sizes
    if total_files < 100:
        volume_category = "small"
        text_chunk_size = 1000
        code_chunk_size = 800
    elif total_files < 500:
        volume_category = "medium"
        text_chunk_size = 750
        code_chunk_size = 500
    else:
        volume_category = "large"
        text_chunk_size = 500
        code_chunk_size = 300
    
    return {
        "total_files": total_files,
        "textual_files": textual_files,
        "code_files": code_files,
        "estimated_tokens": total_tokens,
        "volume_category": volume_category,
        "text_chunk_size": text_chunk_size,
        "code_chunk_size": code_chunk_size
    }


def chunk_by_tokens(text, max_tokens, overlap=50):
    """
    Splits text into chunks based on token count using tiktoken.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    tokens = ENCODING.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = ENCODING.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def get_language_for_file(file_name):
    """
    Maps file extension to LangChain Language enum.
    
    Args:
        file_name: Name of the file
        
    Returns:
        Language enum or None
    """
    extension_map = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TSX,
        ".java": Language.JAVA,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".html": Language.HTML,
        ".css": Language.CSS,
        ".sql": Language.SQL,
        ".sh": Language.BASH,
        ".bash": Language.BASH,
        ".zsh": Language.BASH,
    }
    
    for ext, lang in extension_map.items():
        if file_name.endswith(ext):
            return lang
    return None


def chunk_repository_files(repo_files, max_tokens=None, volume_strategy="auto"):
    """
    Splits file contents into chunks for indexing, separately for textual and code data.
    Uses tiktoken for accurate token counting and LangChain splitters for intelligent chunking.
    
    Args:
        repo_files: List of repository files with their contents
        max_tokens: Maximum number of tokens per chunk (if None, uses volume-aware sizing)
        volume_strategy: Strategy for chunk sizing ("auto", "small", "medium", "large")
        
    Returns:
        Dictionary with chunked textual and code data
    """
    # Analyze repository volume if using auto strategy
    if volume_strategy == "auto" or max_tokens is None:
        volume_info = analyze_repository_volume(repo_files)
        text_chunk_size = volume_info["text_chunk_size"]
        code_chunk_size = volume_info["code_chunk_size"]
        print(f"Repository volume: {volume_info['volume_category']} "
              f"({volume_info['total_files']} files, ~{volume_info['estimated_tokens']:,} tokens)")
        print(f"Using chunk sizes: text={text_chunk_size}, code={code_chunk_size}")
    else:
        text_chunk_size = max_tokens
        code_chunk_size = max_tokens
    
    textual_chunks = []
    code_chunks = []
    
    # Initialize LangChain text splitter for textual content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_chunk_size,
        chunk_overlap=int(text_chunk_size * 0.1),  # 10% overlap
        length_function=lambda text: len(ENCODING.encode(text)),
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Loop through every file in the repository
    for file in repo_files:
        content = file["content"]
        file_name = file["file_name"]
        
        # Skip empty files
        if not content.strip():
            continue
        
        # Determine whether the file is textual or code based on its extension
        if file_name.endswith((".md", ".txt", ".xml", ".rst", ".adoc")):  # Textual data extensions
            # Use LangChain text splitter for better chunking
            text_chunks = text_splitter.split_text(content)
            for idx, chunk in enumerate(text_chunks):
                textual_chunks.append({
                    "file_name": file_name,
                    "content": chunk,
                    "chunk_index": idx
                })
        
        # Code files
        elif file_name.endswith((".py", ".js", ".java", ".html", ".css", ".ts", ".tsx",
                                  ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".rb", ".php",
                                  ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql")):
            # Try to use language-aware splitter
            language = get_language_for_file(file_name)
            
            if language:
                # Use language-specific splitter
                try:
                    code_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=language,
                        chunk_size=code_chunk_size,
                        chunk_overlap=int(code_chunk_size * 0.1),
                        length_function=lambda text: len(ENCODING.encode(text))
                    )
                    code_chunks_split = code_splitter.split_text(content)
                except (AttributeError, TypeError):
                    # Fallback if from_language is not available
                    code_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=code_chunk_size,
                        chunk_overlap=int(code_chunk_size * 0.1),
                        length_function=lambda text: len(ENCODING.encode(text)),
                        separators=["\n\n", "\n", " ", ""]
                    )
                    code_chunks_split = code_splitter.split_text(content)
            else:
                # Fall back to token-based chunking
                code_chunks_split = chunk_by_tokens(content, code_chunk_size)
            
            for idx, chunk in enumerate(code_chunks_split):
                file_ext_match = re.search(r"\.([\w.]+)$", file_name)
                file_extension = file_ext_match.group(1) if file_ext_match else None
                
                code_chunks.append({
                    "file_name": file_name,
                    "file_extension": file_extension,
                    "content": chunk,
                    "chunk_index": idx
                })
    
    return {
        'textual_chunks': textual_chunks,
        'code_chunks': code_chunks
    }
