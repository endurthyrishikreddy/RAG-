import os
import docx
import csv
import fitz


class Loader:
    def __init__(self, upload_dir="uploads_files"):
        os.makedirs(upload_dir, exist_ok=True)
        self.upload_dir = upload_dir

    def load_files(self,file_path: str) -> str:
        "load text content from a file"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        if file_path.endswith('.txt'):
            return self._load_txt(file_path)
        elif file_path.endswith('.csv'):
            return self._load_csv(file_path)    
        elif file_path.endswith('.pdf'):
            return self._load_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}") 


    def _load_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_csv(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                text += ' '.join(row) + '\n'    
            return text.strip()

    def _load_pdf(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + '\n'
        return text.strip()

    def _load_docx(self, file_path: str) -> str:
        document = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        return text.strip()                   