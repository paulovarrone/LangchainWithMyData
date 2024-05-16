from langchain.chains import LLMChain
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.ollama import ChatOllama
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
from texto_pdf_prompt import texto



def pdf_loader_and_splitter(caminho_pdf,conteudo_pasta,BancoVetor,caminho_banco):  
    try:
        # loader = PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\1.pdf")
        # pages = loader.load()
        loaders = [
            PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\base_de_dados\1.pdf"),
            PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\base_de_dados\2.pdf"),
            PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\base_de_dados\3.pdf"),
            PyPDFLoader(r"C:\Users\3470622\Desktop\ChatPdfLocal\base_de_dados\4.pdf")
        ]
        
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        #splitando texto
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

        splits = r_splitter.split_documents(docs)
        
        #string para numeros
        embedding = OllamaEmbeddings(
            base_url="http://localhost:11434", 
            model="mxbai-embed-large",
        )

        #vetorizando
        vectordb = FAISS.from_documents(
            documents=splits,
            embedding=embedding,
            #persist_directory=persist_directory
        )
    

    # salvar banco vetor no diretorio
        try:
            if not BancoVetor:
                os.makedirs(caminho_banco)
                print("Pasta criada com sucesso.")
            else:
                print("Pasta já existe.")
        except Exception as e:
            print(f"ERRO AO CRIAR DIRETÓRIO {e}")

        try:
            vectordb.save_local(folder_path=BancoVetor)
            print("Banco de Vetores pronto")
        except Exception as e:
            print(f"ERRO AO SALVAR VETOR {e}")

    except Exception as e:
       print(f"ERRO AO TENTAR VETORIZAR da função: pdf_loader_and_splitter {e}")
    
    texto_da_inicial = texto(caminho_pdf,conteudo_pasta)

    #-------------------------------------------------------------

    # #comprimindo texto e pegando o mais relevante
    # compressor = LLMChainExtractor.from_llm(chat)

    # #VAMOS USAR MMR???
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor = compressor,
    #     base_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    # )

    # compressed_docs = compression_retriever.get_relevant_documents(PERGUNTA)
    
    # #lamma3 IA
    # chat = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOllama(base_url='http://localhost:11434',
    #     model="llama3:8b", 
    #     temperature=0, 
    #     system='You only speak/write in brazilian portuguese', template= None),
    #     #chain_type=chain_type, 
    #     retriever=compressed_docs, 
    #     return_source_documents=True,
    #     return_generated_question=True,
    # )
    

    # return chat, vectordb