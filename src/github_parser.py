from github import Github

from dotenv import load_dotenv
import os

import re

load_dotenv()
g = Github(os.getenv("GITHUB_API_KEY"))


def get_repo_files(repo_path, file_types=(".md", ".txt", ".py", ".json", ".js", ".html", ".tsx")):
    """
    Parse the repository and extract content from the specified file types.

    Args:
        repo_path: GitHub repository path in the format username/repo_name.
        file_types: Tuple of file extensions to include in the dataset.

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


def chunk_repository_files(repo_files, max_tokens=500):
    """
    Splits file contents into chunks for indexing, separately for textual and code data.
    Args:
        repo_files: List of repository files with their contents
        max_tokens: Maximum number of tokens per chunk

    Returns:
        Two lists: chunked textual data and chunked code data
    """
    textual_chunks = []
    code_chunks = []

    # Loop through every file in the repository
    for file in repo_files:
        content = file["content"]
        file_name = file["file_name"]

        # Determine whether the file is textual or code based on its extension
        if file_name.endswith((".md", ".txt")):  # Textual data extensions
            # Split content into chunks for textual data
            words = content.split()
            for i in range(0, len(words), max_tokens):
                textual_chunks.append({
                    "file_name": file_name,
                    "content": " ".join(words[i:i + max_tokens])
                })

        elif file_name.endswith((".py", ".js", ".java", ".html")):  # Code data extensions
            # Split content into chunks for code data
            words = content.split()
            for i in range(0, len(words), max_tokens):
                code_chunks.append({
                    "file_name": file_name,
                    "file_extension": re.search(r"\.([\w.]+)$", file_name),
                    "content": " ".join(words[i:i + max_tokens])

                })

    return textual_chunks, code_chunks
