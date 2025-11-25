from collections import OrderedDict
from typing import List, Dict, Any

from .metadata_index import MetadataIndex


class LRUCache:
    def __init__(self, capacity: int = 100):
        self.capacity = max(1, capacity)
        self.store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def set(self, key: str, value):
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)


class LazyFileParser:
    """
    On-demand file parser. Uses MetadataIndex to provide shallow parses
    (first N lines) without re-downloading full file contents.
    """

    def __init__(self, metadata: MetadataIndex, cache_size: int = 100):
        self.metadata = metadata
        self.cache = LRUCache(cache_size)

    def parse_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for p in file_paths:
            cached = self.cache.get(p)
            if cached is not None:
                parsed.append(cached)
                continue
            md = self.metadata.by_path.get(p)
            if not md:
                continue
            obj = {
                "path": p,
                "language": md.language,
                "content": md.head,  # shallow parse: head only
                "size": md.size,
                "symbols": md.symbols,
            }
            self.cache.set(p, obj)
            parsed.append(obj)
        return parsed
