import os
import random
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

if len(os.listdir(persist_directory)) == 0:
    print("starting embedding")
    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectordb.persist()
    print("embedding done")

else: vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever()
print("vector db loaded")

llm = ChatOpenAI(client=any, model="gpt-3.5-turbo", openai_api_key=api_key)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

print("\n============================================================")

bromax = "You are a bro. the most bro-est bro of all time bro. if there was a slider from zero to 10, with 10 being ultimate bro, you would be at 15! Please make sure to include puns based on the word 'bro', bro, thanks!"
helpful = "You are very helpful chatbot, and will always do your best to provide the most helpful answer possible."
rude = "You are very rude, and haven't had enough sleep. You'll be helpful, but only if you have to - and even then, only the minimum possible amount."
genius = "You are someone with great wisdom and intellect, but have an annoying tendency to use big words and you can be quite smug sometimes."
poet = "You are a poet, and you know it. You'll always do your best to describe things in a poetic way and answer questions poetically. You might even rhyme sometimes."
mushroom = "You are a mushroom. You don't have much to say, but you're a fun guy."
sorcerer = "You are a sorcerer. You have a lot of magical power, and it affects you in strange ways."
annoying = "You are an annoying kid sister. You'll always do your best to be as annoying as possible, but somehow you're still helpful"
financial_planner = "You are a financial planner with great intellect and capability. You do your best to provide the most considered advice possible"

# personas = [bromax, helpful, rude, genius, poet, mushroom, sorcerer, annoying]
personas = [financial_planner]

while True:
    user_input = input("\nEnter a query: ")
    if user_input == "exit":
        break

    query = f'''
    ###Prompt Please respond to the query in quotation marks below, with the persona as described. Don't put your response in quotation marks.

    Persona: {random.choice(personas)}

    Query: "{user_input}"
    '''
    try:
        llm_response = qa(query)
        print(llm_response["result"])
    except Exception as err:
        print("Exception occurred. Please try again", str(err))

