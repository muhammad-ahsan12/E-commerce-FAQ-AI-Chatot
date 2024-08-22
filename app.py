import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain_community.document_loaders.csv_loader import CSVLoader

# Set up the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"

# Define the file path for the FAISS index
db_file_path = 'FAISS_Index'

# embeddings = HuggingFaceEmbeddings()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)


# Function to create a vector database locally from a given data loader
def creation_of_vectorDB_in_local(loader):
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(db_file_path)

# Function to create a FAQ retrieval chain
def creation_FAQ_chain():
    db = FAISS.load_local(db_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(score_threshold=0.7)

    # Define the prompt template
    prompt_temp = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "This question is not present in my database." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="Question",
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain
# Function to load a CSV file using the CSVLoader
def csv_loader(tmp_file_path):
    loader = CSVLoader(file_path=tmp_file_path)
    return loader

# Load the CSV data and create the vector database
tmp_file_path = "Ecomerese_data.csv"
load = csv_loader(tmp_file_path)
creation_of_vectorDB_in_local(load)
faq_chain = creation_FAQ_chain()

# Streamlit interface
st.title("E-commerce FAQ Retrieval System🛒")
st.markdown("""
<style>
    #MainMenu
    {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
The model is just for E-commerce purpose only. It is trained using  200 Question - Answer pairs.
So, expecting it to answer any question other then used for training with high accuracy is not a good idea.
You can [have a look at all the 200 questions](https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset)
and ask something similar or combined of multiple questions.\n
e.g. ***"How can I create an account?"*** or,\n
***"What payment methods do you accept?"*** or even,\n
***"How can I create an account and what payment methods do you accept?"*** , etc.\n
you found this project [on my Github](https://github.com/muhammad-ahsan12/Ecomerse-Chatbot.git)
""", unsafe_allow_html=True)
query = st.text_input("Enter your question:")
if st.button("🔮Get Answer"):
    with st.spinner("🔄processing..."):
        result = faq_chain(query)
        if 'result' in result:
            st.write(result['result'])
        else:
            st.write("Answer not found.")

