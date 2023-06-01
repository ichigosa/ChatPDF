import os
from langchain.document_loaders import PyMuPDFLoader

OpenAI_API_Key = os.environ['OPENAI_API_KEY']
print(OpenAI_API_Key)

loader = PyMuPDFLoader("./docs/example.pdf")
documents = loader.load()