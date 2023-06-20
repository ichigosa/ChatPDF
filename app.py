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
print("document loading done")

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
texts = text_splitter.split_documents(documents)
print("text split done")

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
print("embedding done")

if len(os.listdir(persist_directory)) == 0:
    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectordb.persist()

else: vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever()
print("vector db loaded")

llm = ChatOpenAI(client=any, model="gpt-3.5-turbo", openai_api_key=api_key)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

print("\n============================================================")

while True:
    user_input = input("\nEnter a query: ")
    if user_input == "exit":
        break

    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        print(llm_response["result"])
    except Exception as err:
        print("Exception occured. Please try again", str(err))

