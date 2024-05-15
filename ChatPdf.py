import __main__
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
import fitz

def pdf_loader_and_splitter():  
    try:
        # loader = PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\1.pdf")
        # pages = loader.load()
        loaders = [
            PyPDFLoader("./1.pdf"),
            PyPDFLoader("./2.pdf"),
            PyPDFLoader("./3.pdf"),
            PyPDFLoader("./4.pdf")
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
            #temperature=0
        )

        #vetorizando
        vectordb = FAISS.from_documents(
            documents=splits,
            embedding=embedding,
            #persist_directory=persist_directory
        )

        # salvar banco vetor no diretorio
        # vectordb.save_local(folder_path="./BancoVetor/")
        # print("Banco de Vetores pronto")
    except Exception as e:
       print(f"ERRO AO TENTAR VETORIZAR da função: pdf_loader_and_splitter {vectordb}: {e}")

    #-------------------------------------------------------------

    #comprimindo texto e pegando o mais relevante
    compressor = LLMChainExtractor.from_llm(chat)

    #VAMOS USAR MMR???
    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    )

    compressed_docs = compression_retriever.get_relevant_documents(PERGUNTA)
    
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



def extract_text_from_pdf(caminho_pdf):
  try:

    text = ""

    with fitz.open(caminho_pdf) as pdf_file:
      for page in pdf_file: 
        text += page.get_text()

    return text
  
  except Exception as e:
    print(f"ERRO AO TENTAR EXTRAIR TEXTO da função: extract_text_from_pdf {text}: {e}")


def texto(caminho_pdf):
   
   conteudo_pasta = os.listdir(caminho_pdf)
   
   try:
    for arquivo in conteudo_pasta:
        if arquivo.endswith('.pdf'):
            pdf_file = os.path.join(caminho_pdf, arquivo)
            texto_pdf = extract_text_from_pdf(pdf_file)

    return texto_pdf
   
   except Exception as e:
      print(f"ERRO AO TENTAR EXTRAIR TEXTO da função: texto {texto_pdf}: {e}")


def main():
   try:
       caminho_pdf = './pdf'

   except Exception as e:
       print(f"Erro na função: main  {caminho_pdf}: {e}")

if __name__ is "__main__":
   main()