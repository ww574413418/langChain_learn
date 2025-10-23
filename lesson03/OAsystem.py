from dotenv import load_dotenv
import os

# load api key
load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# 1.load data
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader

# file document path
base_dir = os.path.join(os.path.dirname(__file__), "oneFlowers")
documents = []

for file in os.listdir(base_dir):
    # build whole file path
    file_path = os.path.join(base_dir,file)

    if file.endswith(".doc"):
        loader = Docx2txtLoader(file_path)
        documents.append(loader.load())
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.append(loader.load())
    elif file.endswith(".text"):
        loader = TextLoader(file_path)
        documents.append(loader.load())


# 2.split text
from langchain.text_splitter import RecursiveCharacterTextSplitter
# text has been divide into 200 shots
text_spliter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)


flat_documents = [doc for sublist in documents for doc in sublist]

# using text_spliter to slit text
chunked_documents = text_spliter.split_documents(flat_documents)

# 3.Store Embed and segment the data and store it in the vector database Qdrant.
from langchain_community.vectorstores import Qdrant

from langchain.pydantic_v1 import BaseModel
from langchain.embeddings.base import Embeddings

class DoubaoEmbeddings(BaseModel, Embeddings):
    client: Ark = None
    api_key: str = ""
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.api_key == "":
            self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = Ark(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Generate the embedding of the input text.
        Args:
            texts (str): to generate embedding text
        Return:
            embeddings (List[float]): embedding，a float value list.
        """
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True


vectorstore = Qdrant.from_documents(
    # splited document
    documents=chunked_documents,
    embedding=DoubaoEmbeddings(
        model=os.environ["EMBEDDING_MODELEND"],
    ),
    # using OpenAI`s Embedding Model
    location=":memory:",
    # in-memory
    collection_name="my_documents",
)

