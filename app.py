import os
from langchain.document_loaders import PyMuPDFLoader

openai.api_key = os.environ['OPENAI_AI_KEY']

loader = PyMuPDFLoader(/path/to/pdf)
documents = loader.load()