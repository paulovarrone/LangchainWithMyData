import os
from vetorizar import pdf_loader_and_splitter


def main():
   try:
       caminho_pdf = r'C:\Users\3470622\Desktop\ChatPdfLocal\pdf'
       caminho_banco = r'C:\Users\3470622\Desktop\ChatPdfLocal\BancoVetor'
       conteudo_pasta = os.listdir(caminho_pdf)
       BancoVetor = os.path.exists(caminho_banco)

       pdf_loader_and_splitter(caminho_pdf,conteudo_pasta,BancoVetor,caminho_banco)
       

   except Exception as e:
       print(f"Erro na função: main  {e}")

if __name__ == "__main__":
   main()