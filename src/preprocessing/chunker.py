import os
import json
from typing import List, Dict, Any
import fitz  # PyMuPDF


class Chunker:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.chunks_dir = os.path.join(data_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or TXT file"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_from_pdf(file_path)
        elif ext == ".txt":
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")

    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        return text

    def _extract_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def save_chunks(self, chunks: List[str], filename: str) -> str:
        """Save chunks to JSON file"""
        chunks_data = {
            "source_file": filename,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": i,
                    "content": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
        }

        chunks_filename = f"{os.path.splitext(filename)[0]}_chunks.json"
        chunks_path = os.path.join(self.chunks_dir, chunks_filename)

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        return chunks_path

    def load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file"""
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["chunks"]

    def process_file(self, file_path: str, chunk_size: int = 1000, overlap: int = 100) -> str:
        """Process PDF or TXT file and save chunks"""
        print(f"Processing: {file_path}")

        # Extract text
        text = self.extract_text(file_path)
        if not text:
            raise ValueError(f"No text extracted from {file_path}")

        print(f"Extracted {len(text)} characters")

        # Chunk text
        chunks = self.chunk_text(text, chunk_size, overlap)
        print(f"Created {len(chunks)} chunks")

        # Save chunks
        filename = os.path.basename(file_path)
        chunks_path = self.save_chunks(chunks, filename)
        print(f"Saved chunks to: {chunks_path}")

        return chunks_path


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    CONFIG = {
        # Source file (.pdf or .txt)
        "file_path": "data/documents/document_handbook_mei_2024.pdf",

        # Chunking Parameters
        "chunk_size": 500,  # Number of words per chunk
        "overlap": 100,  # Number of overlapping words between chunks
    }
    # ========================

    chunker = Chunker()

    if os.path.exists(CONFIG["file_path"]):
        chunks_path = chunker.process_file(
            CONFIG["file_path"],
            chunk_size=CONFIG["chunk_size"],
            overlap=CONFIG["overlap"]
        )
        print(f"\nChunks saved to: {chunks_path}")
    else:
        print(f"File not found: {CONFIG['file_path']}")
