from typing import List, Dict, Any


class MetadataFilter:
    def __init__(self, metadata_store:List[Dict[str, Any]]):
        self.metadata_store = metadata_store

    def filter(self, docs: List[str], filter_query: Dict[str, Any]):
        """
        filter_query example:
        { "source": "manual.pdf" }
        """
        results = []
        for idx, doc in enumerate(docs):
            meta = self.metadata[idx]
            # Check if ALL keys match
            if all(meta.get(k) == v for k, v in filter_query.items()):
                results.append(doc)

        return results