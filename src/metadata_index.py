import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .github_parser import get_repo_files, get_language_for_file


@dataclass
class FileMetadata:
    path: str
    size: int
    mtime: Optional[float]  # unknown for GitHub API, reserved for local repos
    language: str
    head: str  # first N lines joined
    symbols: List[str] = field(default_factory=list)


class MetadataIndex:
    """
    Lightweight in-memory metadata index for a repository.
    Contains only minimal information required for fast candidate selection.
    """

    def __init__(self, repo_path: str, head_lines: int = 20):
        self.repo_path = repo_path
        self.head_lines = head_lines
        self.by_path: Dict[str, FileMetadata] = {}
        # Simple reverse index for quick keyword/path matching
        self.symbol_to_paths: Dict[str, List[str]] = {}

    def build(self) -> "MetadataIndex":
        """
        Build the metadata index using an existing lightweight repository fetcher.
        Note: For GitHub-based repos via PyGithub, we currently fetch file contents
        through get_repo_files for simplicity, but we only keep the first N lines
        to minimize memory usage. This avoids heavy embedding/vector DB work at startup.
        """
        files = get_repo_files(self.repo_path)
        for f in files:
            path = f.get("file_name", "")
            content = f.get("content", "")
            head = self._first_lines(content, self.head_lines)
            size = len(content.encode("utf-8", errors="ignore"))
            lang_enum = get_language_for_file(path)
            lang_name = self._language_name(path, lang_enum)
            symbols = self._extract_symbols(lang_name, head)
            md = FileMetadata(
                path=path,
                size=size,
                mtime=None,
                language=lang_name,
                head=head,
                symbols=symbols,
            )
            self.by_path[path] = md
            for s in symbols:
                self.symbol_to_paths.setdefault(s.lower(), []).append(path)
        return self

    def _first_lines(self, content: str, n: int) -> str:
        lines = content.splitlines()[:n]
        return "\n".join(lines)

    def _extract_symbols(self, language: str, text: str) -> List[str]:
        """
        Quick, regex-based symbol extraction to avoid full parsing.
        Supports common patterns across languages.
        """
        symbols: List[str] = []
        try:
            if (language or "").lower() == "python":
                # def func(...): or class Name(...):
                symbols += re.findall(r"^\s*def\s+([a-zA-Z_][\w]*)\s*\(", text, flags=re.MULTILINE)
                symbols += re.findall(r"^\s*class\s+([a-zA-Z_][\w]*)\s*\(", text, flags=re.MULTILINE)
            elif (language or "").lower() in {"javascript", "typescript"}:
                # function name( or class Name {
                symbols += re.findall(r"function\s+([a-zA-Z_][\w]*)\s*\(", text)
                symbols += re.findall(r"class\s+([A-Za-z_][\w]*)\b", text)
                symbols += re.findall(r"const\s+([a-zA-Z_][\w]*)\s*=\s*\(", text)
            elif (language or "").lower() in {"java", "kotlin", "scala", "swift"}:
                symbols += re.findall(r"class\s+([A-Za-z_][\w]*)\b", text)
                symbols += re.findall(r"interface\s+([A-Za-z_][\w]*)\b", text)
            elif (language or "").lower() in {"go"}:
                symbols += re.findall(r"^\s*func\s+(?:\(.*?\)\s*)?([A-Za-z_][\w]*)\s*\(", text, flags=re.MULTILINE)
            else:
                # Generic heuristics
                symbols += re.findall(r"class\s+([A-Za-z_][\w]*)\b", text)
                symbols += re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(.*?\)\s*{", text)
        except Exception:
            pass
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for s in symbols:
            sl = s.lower()
            if sl not in seen:
                seen.add(sl)
                uniq.append(s)
        return uniq

    def list_paths(self) -> List[str]:
        return list(self.by_path.keys())

    def search_by_symbol(self, token: str) -> List[str]:
        return self.symbol_to_paths.get(token.lower(), [])

    def _language_name(self, path: str, lang_enum) -> str:
        """Convert LangChain Language enum (or None) and path to a readable language name."""
        # If enum-like with name attribute
        try:
            if hasattr(lang_enum, "name") and lang_enum.name:
                name = str(lang_enum.name).lower()
            elif lang_enum:
                name = str(lang_enum).lower()
            else:
                name = "unknown"
        except Exception:
            name = "unknown"

        # Map common enum names to canonical language names
        mapping = {
            "python": "python",
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rust": "rust",
            "ruby": "ruby",
            "php": "php",
            "swift": "swift",
            "kotlin": "kotlin",
            "scala": "scala",
            "html": "html",
            "css": "css",
            "sql": "sql",
            "bash": "bash",
        }
        if name in mapping:
            return mapping[name]

        # Fallback by extension
        lower = path.lower()
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".sh": "bash",
            ".md": "markdown",
            ".txt": "text",
            ".rst": "text",
            ".adoc": "text",
        }
        for ext, nm in ext_map.items():
            if lower.endswith(ext):
                return nm
        return name or "unknown"
