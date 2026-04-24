import os
from operator import index

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
#from ollama import embeddings

load_dotenv(override=True)


def main():
    # STEP 1: -  Document loading
    print("Loading OpenAI embeddings...")
    loader = TextLoader("./rag/mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    # Step 2: - chunking ( splitting text)
    print("splitting text")
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"texts len {len(texts)}")
    print(f"text[1] {texts[0]}")

    # Step 3:-embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Step 4: Store embeddings in vector database
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")






#  uv run .\rag\ingestion.py
if __name__ == "__main__":
    main()


