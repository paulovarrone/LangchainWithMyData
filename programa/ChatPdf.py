import os

def main():
   try:
       caminho_pdf = r'C:\Users\3470622\Desktop\ChatPdfLocal\pdf'
       caminho_banco = r'C:ChatPdfLocal\base_de_dados'
       BancoVetor = os.path.exists(caminho_banco)

   except Exception as e:
       print(f"Erro na função: main  {caminho_pdf}: {e}")

if __name__ == "__main__":
   main()