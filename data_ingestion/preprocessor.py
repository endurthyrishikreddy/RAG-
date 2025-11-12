import re
from typing import List


class TextPreprocessor:
    def __init__(self,chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        "Clean the text by removing extra whitespace and special characters"
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
        text = text.replace('\n', ' ').strip()  # Replace newlines with a space    
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        "Split the text into chunks with specified size and overlap"
        cleaned_text = self.clean_text(text)
        chunks = []
        start = 0
        while start < len(cleaned_text):
            end = start + self.chunk_size
            chunk = cleaned_text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap  # Move start forward by chunk size minus overlap
        return chunks
