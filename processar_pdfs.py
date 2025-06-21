import os
import json
import PyPDF2
import google.generativeai as genai
import streamlit as st
import time

# --- CONFIGURAÇÕES ---
PASTA_COMUNICADOS = "comunicados"
ARQUIVO_SAIDA_JSON = "base_conhecimento.json"
SEGMENTOS_SIGLAS = ["AI", "AF", "EI", "EM"]
MODELO_EMBEDDING = 'models/text-embedding-004'

# --- CONFIGURAÇÃO DA API ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    print("Chave da API carregada do arquivo secrets.toml do Streamlit.")
except (KeyError, AttributeError, FileNotFoundError):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Chave da API não encontrada na variável de ambiente GEMINI_API_KEY.")
        genai.configure(api_key=api_key)
        print("Chave da API carregada da variável de ambiente.")
    except Exception as e:
        print(f"🚨 Erro ao configurar a API do Gemini: {e}")
        print("Certifique-se de que sua chave GEMINI_API_KEY está no arquivo .streamlit/secrets.toml ou definida como uma variável de ambiente.")
        exit()

# --- FUNÇÕES (sem alterações) ---
def extrair_texto_pdf(caminho_pdf):
    try:
        with open(caminho_pdf, 'rb') as arquivo:
            leitor_pdf = PyPDF2.PdfReader(arquivo)
            return "\n".join(page.extract_text() for page in leitor_pdf.pages if page.extract_text())
    except Exception as e:
        print(f"Erro ao ler {caminho_pdf}: {e}")
        return None

def identificar_segmentos(nome_arquivo):
    segmentos_encontrados = []
    nome_sem_ext = nome_arquivo.lower().replace('.pdf', '')
    nome_padded = f" {nome_sem_ext} "
    for sigla in SEGMENTOS_SIGLAS:
        sigla_lower = sigla.lower()
        if f" {sigla_lower} " in nome_padded or f"-{sigla_lower}-" in nome_padded or f" {sigla_lower}(" in nome_padded or f"_{sigla_lower}_" in nome_padded or f" {sigla_lower}." in nome_padded:
            segmentos_encontrados.append(sigla)
    segmentos_unicos = list(set(segmentos_encontrados))
    return segmentos_unicos if segmentos_unicos else ["Geral"]

def chunk_texto(texto, tamanho_chunk=2000, sobreposicao=200):
    if not texto: return []
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho_chunk
        chunks.append(texto[inicio:fim])
        inicio += tamanho_chunk - sobreposicao
    return chunks

# --- FUNÇÃO PRINCIPAL (COM A NOVA LÓGICA) ---
def processar_pasta_comunicados():
    """
    Orquestra o processo de forma incremental: lê a base de dados existente,
    identifica apenas os novos PDFs e processa somente eles.
    """
    # 1. Carrega a base de conhecimento existente, se houver
    base_conhecimento_vetorial = []
    arquivos_ja_processados = set()
    if os.path.exists(ARQUIVO_SAIDA_JSON):
        print(f"Carregando base de conhecimento existente de '{ARQUIVO_SAIDA_JSON}'...")
        with open(ARQUIVO_SAIDA_JSON, 'r', encoding='utf-8') as f:
            base_conhecimento_vetorial = json.load(f)
        arquivos_ja_processados = {doc['arquivo'] for doc in base_conhecimento_vetorial}
        print(f"{len(arquivos_ja_processados)} arquivos já foram processados.")

    # 2. Identifica apenas os arquivos novos na pasta
    print(f"Verificando a pasta '{PASTA_COMUNICADOS}' por novos arquivos...")
    if not os.path.isdir(PASTA_COMUNICADOS):
        print(f"Erro: Pasta '{PASTA_COMUNICADOS}' não encontrada.")
        return

    todos_os_arquivos_na_pasta = {f for f in os.listdir(PASTA_COMUNICADOS) if f.lower().endswith(".pdf")}
    arquivos_novos = list(todos_os_arquivos_na_pasta - arquivos_ja_processados)

    if not arquivos_novos:
        print("\nNenhum arquivo novo para processar. A base de conhecimento já está atualizada. ✨")
        return

    print(f"\nEncontrados {len(arquivos_novos)} novos arquivos para processar:")
    for arq in arquivos_novos: print(f" - {arq}")

    # 3. Processa APENAS os arquivos novos
    for i, nome_arquivo in enumerate(arquivos_novos):
        print(f"\nProcessando novo arquivo {i+1}/{len(arquivos_novos)}: {nome_arquivo}...")
        caminho_completo = os.path.join(PASTA_COMUNICADOS, nome_arquivo)
        
        texto_completo = extrair_texto_pdf(caminho_completo)
        if not texto_completo:
            continue

        segmentos = identificar_segmentos(nome_arquivo)
        print(f"  - Segmentos identificados: {segmentos}")
        chunks_de_texto = chunk_texto(texto_completo)
        
        documento_processado = {"arquivo": nome_arquivo, "segmentos": segmentos, "chunks": []}
        
        for j, chunk in enumerate(chunks_de_texto):
            try:
                time.sleep(1) 
                print(f"    - Gerando embedding para o chunk {j+1}/{len(chunks_de_texto)}...")
                embedding_result = genai.embed_content(model=MODELO_EMBEDDING, content=chunk)
                documento_processado["chunks"].append({"texto_chunk": chunk, "vetor": embedding_result['embedding']})
            except Exception as e:
                print(f"    !! Erro ao gerar embedding para o chunk: {e}. Pulando este chunk.")

        base_conhecimento_vetorial.append(documento_processado)

    # 4. Salva a base de dados combinada
    print("-" * 20)
    print("Salvando a base de conhecimento atualizada...")
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(base_conhecimento_vetorial, f, ensure_ascii=False, indent=2)
    
    print("Processamento concluído com sucesso!")

if __name__ == "__main__":
    processar_pasta_comunicados()