from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def load_and_split_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Total pages loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def upload_to_pinecone(chunks):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(PINECONE_INDEX_NAME)

    records = []

    for i, chunk in enumerate(chunks):
        records.append({
            "_id": f"chunk-{i}",
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", 0)
        })

    batch_size = 90

    print(f"Total records to upload: {len(records)}")

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        print(
            f"Uploading records {start + 1} "
            f"to {start + len(batch)}..."
        )

        index.upsert_records(
            namespace="agentic-ai",
            records=batch
        )

    print(f"Successfully uploaded {len(records)} chunks to Pinecone!")


if __name__ == "__main__":
    chunks = load_and_split_pdf()
    upload_to_pinecone(chunks)