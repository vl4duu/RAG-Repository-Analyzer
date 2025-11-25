import os
import re
from typing import List, Dict, Tuple

from .metadata_index import MetadataIndex, FileMetadata


class FileSelector:
    """
    Query-driven file selector. Uses lightweight heuristics:
    - keyword matching against file paths, head content, and extracted symbols
    - simple path pattern hints
    - naive dependency expansion based on import/include statements
    """

    def __init__(self, metadata: MetadataIndex):
        self.metadata = metadata

    def select_files(self, query: str, max_files: int = 20) -> List[str]:
        tokens = self._extract_keywords(query)
        scores: Dict[str, float] = {}

        # Path and symbol matches
        for path, md in self.metadata.by_path.items():
            score = 0.0
            path_l = path.lower()
            head_l = md.head.lower()

            for t in tokens:
                if t in path_l:
                    score += 2.0
                if t in head_l:
                    score += 1.0
            # symbol exact/partial
            for sym in md.symbols:
                sl = sym.lower()
                if sl in tokens:
                    score += 3.0
                else:
                    for t in tokens:
                        if t in sl:
                            score += 1.5
            # path pattern hints
            score += self._path_hints(tokens, path_l)

            if score > 0:
                scores[path] = scores.get(path, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [p for p, _ in ranked[:max_files]]

        # Simple dependency expansion for Python/TS: include files that are imported by or import selected ones
        expanded = set(candidates)
        for p in list(candidates):
            deps = self._dependencies(p)
            for d in deps:
                if len(expanded) < max_files:
                    expanded.add(d)
                else:
                    break
        return list(expanded)[:max_files]

    def _extract_keywords(self, query: str) -> List[str]:
        q = query.lower()
        # crude noun/keyword extraction: alphanumerics of length >= 3
        words = re.findall(r"[a-z0-9_]{3,}", q)
        return list(
            {w for w in words if w not in {"what", "this", "about", "that", "with", "from", "repo", "repository"}})

    def _path_hints(self, tokens: List[str], path_l: str) -> float:
        hints = {
            "auth": ["auth", "security", "login", "oauth"],
            "test": ["test", "tests", "spec"],
            "api": ["api", "endpoint", "router"],
            "db": ["db", "database", "model", "schema"],
            "docs": ["readme", "docs", "guide", "tutorial"],
        }
        score = 0.0
        for key, kws in hints.items():
            if any(t in tokens for t in kws) and key in path_l:
                score += 1.0
        return score

    def _dependencies(self, path: str) -> List[str]:
        md = self.metadata.by_path.get(path)
        if not md:
            return []
        deps = []
        head = md.head
        # Python imports
        for m in re.findall(r"^\s*import\s+([a-zA-Z0-9_\.]+)|^\s*from\s+([a-zA-Z0-9_\.]+)\s+import", head,
                            re.MULTILINE):
            mod = (m[0] or m[1]).split(".")[0]
            # map module to file path best-effort
            guess = f"{mod}.py"
            for p in self.metadata.by_path.keys():
                if p.endswith(guess) or f"/{guess}" in p:
                    deps.append(p)
        # JS/TS imports
        for m in re.findall(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", head):
            rel = m
            if rel.startswith("."):
                base_dir = os.path.dirname(path)
                for ext in (".ts", ".tsx", ".js", ".jsx"):
                    candidate = os.path.normpath(os.path.join(base_dir, rel + ext))
                    for p in self.metadata.by_path.keys():
                        if p.endswith(candidate) or p.endswith(os.path.basename(candidate)):
                            deps.append(p)
        return deps
