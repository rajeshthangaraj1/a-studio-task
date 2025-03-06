import os
import hashlib
import io
import pandas as pd
import json
from PyPDF2 import PdfReader
from docx import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FileHandler:
    def __init__(self,vector_db_path,api_token,logger):
        self.logger = logger
        self.logger.info("Initializing FileHandler...")
        self.vector_db_path = vector_db_path
        # Initialize the embedding model using Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": api_token},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
    def handle_file_upload(self, file, document_name, document_description):
        try:
            content = file.read()
            file_hash = hashlib.md5(content).hexdigest()
            normalized_filename = file.name.lower().replace(' ', '_')  # ✅ Fix here
            file_key = f"{normalized_filename}_{file_hash}"
            vector_store_dir = os.path.join(self.vector_db_path, file_key)
            os.makedirs(vector_store_dir, exist_ok=True)
            vector_store_path = os.path.join(vector_store_dir, "index.faiss")

            if os.path.exists(vector_store_path):
                return {"message": "File already processed."}

            # Process file based on type
            if file.name.endswith(".pdf"):
                texts, metadatas = self.load_and_split_pdf(file)
            elif file.name.endswith(".docx"):
                texts, metadatas = self.load_and_split_docx(file)
            elif file.name.endswith(".txt"):
                texts, metadatas = self.load_and_split_txt(content)
            elif file.name.endswith(".xlsx"):
                texts, metadatas = self.load_and_split_table(content)
            elif file.name.endswith(".csv"):
                texts, metadatas = self.load_and_split_csv(content)
            else:
                raise ValueError("Unsupported file format.")

            if not texts:
                return {"message": "No text extracted from the file. Check the file content."}


            # Apply chunking to split extracted text into smaller pieces
            chunked_texts = []
            chunked_metadatas = []
            for text_chunk, metadata in zip(texts, metadatas):
                chunks = self.text_splitter.split_text(text_chunk)
                chunked_texts.extend(chunks)
                chunked_metadatas.extend([metadata] * len(chunks))  # Assign same metadata to chunks

            # Create FAISS vector store from chunked text
            vector_store = FAISS.from_texts(chunked_texts, self.embeddings, metadatas=chunked_metadatas)
            vector_store.save_local(vector_store_dir)

            metadata = {
                "filename": file.name,
                "document_name": document_name,
                "document_description": document_description,
                "file_size": len(content),
            }
            metadata_path = os.path.join(vector_store_dir, "metadata.json")
            with open(metadata_path, 'w') as md_file:
                json.dump(metadata, md_file)

            return {"message": "File processed successfully."}
        except Exception as e:
            return {"message": f"Error processing file: {str(e)}"}
    def load_and_split_pdf(self, file):
        reader = PdfReader(file)
        texts = []
        metadatas = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
                metadatas.append({"page_number": page_num + 1})
        return texts, metadatas

    def load_and_split_docx(self, file):
        doc = Document(file)
        texts = []
        metadatas = []
        for para_num, paragraph in enumerate(doc.paragraphs):
            if paragraph.text:
                texts.append(paragraph.text)
                metadatas.append({"paragraph_number": para_num + 1})
        return texts, metadatas

    def load_and_split_txt(self, content):
        text = content.decode("utf-8")
        lines = text.split('\n')
        texts = [line for line in lines if line.strip()]
        metadatas = [{}] * len(texts)
        return texts, metadatas

    def load_and_split_table(self, content):
        excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
        texts = []
        metadatas = []
        for sheet_name, df in excel_data.items():
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            df = df.fillna('N/A')
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Combine key-value pairs into a string
                row_text = ', '.join([f"{key}: {value}" for key, value in row_dict.items()])
                texts.append(row_text)
                metadatas.append({"sheet_name": sheet_name})
        return texts, metadatas

    def load_and_split_csv(self, content):
        csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        texts = []
        metadatas = []
        csv_data = csv_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        csv_data = csv_data.fillna('N/A')
        for _, row in csv_data.iterrows():
            row_dict = row.to_dict()
            row_text = ', '.join([f"{key}: {value}" for key, value in row_dict.items()])
            texts.append(row_text)
            metadatas.append({"row_index": _})
        return texts, metadatas

