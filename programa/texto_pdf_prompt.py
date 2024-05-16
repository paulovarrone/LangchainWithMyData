import os
from extrair_texto import extract_text_from_pdf

def texto(caminho_pdf,conteudo_pasta):
   
   try:
    for arquivo in conteudo_pasta:
        if arquivo.endswith('.pdf'):
            pdf_file = os.path.join(caminho_pdf, arquivo)
            texto_pdf = extract_text_from_pdf(pdf_file)

    return texto_pdf
   
   except Exception as e:
      print(f"ERRO AO TENTAR EXTRAIR TEXTO da função: texto {e}")