import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

persist_directory = "./storage"
pdf_path = "pdf/test.pdf"
api_key = os.environ["OPENAI_API_KEY"]

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(searcg_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=api_key)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    user_input = input("Enter a query: ")
    if user_input == "exit":
        break

    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        print(llm_response["result"])
    except Exception as err:
        print("Exception occured. Please try again", str(err))

