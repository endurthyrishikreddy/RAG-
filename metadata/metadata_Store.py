import json
import os
from typing import List, Dict,Any
import uuid
from datetime import datetime

class MetadataStore:
    def __init__(self, file_path: str = "vectore_store/metadata.json"):
        self.file_path = file_path
        self.data = self.load()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.load()


    def load(self):
        "Load metadata from the JSON file"
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: Metadata file is corrupted. Starting with an empty store.")
                self.data = {}

        else:
            print("ℹ️ No existing metadata file found. Starting with an empty store.")
            self.data = {}

    def save(self):
        "Save metadata to the JSON file"
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def add_document(self, filename: str, num_chunks: int, path: str, start_idx: int, end_idx: int) -> str:  
        "Add a new document's metadata and return its unique ID"
        doc_id = str(uuid.uuid4())
        self.data[doc_id] = {
            "filename": filename,
            "num_chunks": num_chunks,
            "path": path,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamp": datetime.now().isoformat()
        }
        self.save()
        return doc_id
    
    def list_documents(self) -> List[Dict[str, Any]]:
        "List all documents in the metadata store"
        return [{"id": doc_id, **info} for doc_id, info in self.data.items()]  

    def delete_document(self, doc_id: str):
        "Delete a document's metadata by its ID"
        if doc_id in self.data:
            del self.data[doc_id]
            self.save()

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        "Retrieve a document's metadata by its ID"
        return self.data.get(doc_id, {})


             
    


    