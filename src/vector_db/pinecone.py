import os
from typing import Any, List

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv(".env")

PC = Pinecone(api_key= st.secrets.api_key)
PC_INDEX_NAME = st.secrets.pc_index_name

EMBEDDINNGS = AzureOpenAIEmbeddings(
    azure_endpoint=st.secrets.azure_endpoint,
    api_key= st.secrets.api_key,
    azure_deployment=st.secrets.azure_deploment,
    openai_api_version=st.secrets.openai_api_version
)

class PineconeDB:
    def __init__(self) -> None:
        self._create_index()
        self.index = PC.Index(PC_INDEX_NAME)
        self.vectorstore = PineconeVectorStore(
            index_name=PC_INDEX_NAME,
            embedding=EMBEDDINNGS,
            pinecone_api_key = st.secrets.pinecone_api_key
        )

    def _create_index(self, *args, **kwarg) -> None:
        # check if index exists
        existing_indexes = [
            index_info["name"] for index_info in PC.list_indexes()
        ]
        if PC_INDEX_NAME in existing_indexes:
            return
        PC.create_index(
            name=PC_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    def ingest_data(self, docs: List[Any]) -> None:
        self.vectorstore = PineconeVectorStore.from_documents(
            docs, EMBEDDINNGS, index_name=PC_INDEX_NAME
        )

    def add_data(self, *args, **kwarg) -> None:
        self.vectorstore.add_documents(*args, **kwarg)

    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        return self.vectorstore.similarity_search(query, k=k)

    def vectorstore_retriever(self, *args, **kwarg) -> Any:
        # documentation: https://api.python.langchain.com/en/latest/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html#langchain_pinecone.vectorstores.PineconeVectorStore.as_retriever
        return self.vectorstore.as_retriever(*args, **kwarg)

    def delete_index(self, name: str) -> None:
        PC.delete_index(name)
