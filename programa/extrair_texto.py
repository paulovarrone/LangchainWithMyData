import fitz

def extract_text_from_pdf(caminho_pdf):
  try:

    text = ""

    with fitz.open(caminho_pdf) as pdf_file:
      for page in pdf_file: 
        text += page.get_text()
      print('Texto extraido com sucesso na function extract_text_from_pdf()')  
    return text
    
  except Exception as e:
    print(f"ERRO AO TENTAR EXTRAIR TEXTO da função: extract_text_from_pdf {text}: {e}")