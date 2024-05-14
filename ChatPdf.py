import os
import datetime
from langchain.chains import LLMChain
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models.ollama import ChatOllama
# from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
#EXTRA
from langchain_community.llms.ollama import Ollama

def pdf_loader_and_splitter():  
    # loader = PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\1.pdf")
    # pages = loader.load()
    loaders = [
        PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\1.pdf"),
        PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\2.pdf"),
        PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\3.pdf"),
        PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\4.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    #splitando texto
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    splits = r_splitter.split_documents(docs)
    
    #string para numeros
    embedding = OllamaEmbeddings(
        base_url="http://localhost:11434", 
        model="mxbai-embed-large",
        temperature=0
    )

    #vetorizando
    vectordb = FAISS.from_documents(
        documents=splits,
        embedding=embedding,
        #persist_directory=persist_directory
    )


    #comprimindo texto e pegando o mais relevante
    compressor = LLMChainExtractor.from_llm(chat)

    #VAMOS USAR MMR???
    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    )

    compressed_docs = compression_retriever.get_relevant_documents("PERGUNTA")
    
    #lamma3 IA
    chat = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(base_url='http://localhost:11434',
        model="llama3:8b", 
        temperature=0, 
        system='You only speak/write in brazilian portuguese'), 
        #chain_type=chain_type, 
        retriever=compressed_docs, 
        return_source_documents=True,
        return_generated_question=True,
    )
    

    return chat, vectordb


def template(vectordb, chat):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(input_variables=["context", "question"],template=template)

    qa_chain = RetrievalQA.from_chain_type(
        chat,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain