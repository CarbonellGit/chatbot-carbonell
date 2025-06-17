import os
import json
import PyPDF2
import google.generativeai as genai
import streamlit as st # Usamos o Streamlit para carregar os secrets
import time

# --- CONFIGURAÇÕES ---
PASTA_COMUNICADOS = "comunicados"
ARQUIVO_SAIDA_JSON = "base_conhecimento.json"
SEGMENTOS_SIGLAS = ["AI", "AF", "EI", "EM"] # Siglas para Anos Iniciais, Finais, Ensino Infantil e Médio
MODELO_EMBEDDING = 'models/text-embedding-004'

# --- CONFIGURAÇÃO DA API ---
# Carrega a chave da API dos segredos do Streamlit ou variáveis de ambiente
try:
    # Tenta carregar do Streamlit Secrets primeiro
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    print("Chave da API carregada do arquivo secrets.toml do Streamlit.")
except (KeyError, AttributeError, FileNotFoundError):
    # Se falhar, tenta carregar de uma variável de ambiente (útil para rodar localmente sem o app Streamlit)
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


# --- FUNÇÕES ---

def extrair_texto_pdf(caminho_pdf):
    """Extrai texto de um arquivo PDF."""
    try:
        with open(caminho_pdf, 'rb') as arquivo:
            leitor_pdf = PyPDF2.PdfReader(arquivo)
            return "\n".join(page.extract_text() for page in leitor_pdf.pages if page.extract_text())
    except Exception as e:
        print(f"Erro ao ler {caminho_pdf}: {e}")
        return None

def identificar_segmentos(nome_arquivo):
    """
    Identifica segmentos a partir do nome do arquivo de forma mais robusta,
    procurando pela sigla cercada por delimitadores comuns.
    """
    segmentos_encontrados = []
    # Remove a extensão .pdf e adiciona espaços no início e fim para facilitar a busca
    nome_sem_ext = nome_arquivo.lower().replace('.pdf', '')
    nome_padded = f" {nome_sem_ext} "

    for sigla in SEGMENTOS_SIGLAS:
        sigla_lower = sigla.lower()
        # Procura por padrões como " ai ", "-ai-", " ai(", " ai_", etc.
        if f" {sigla_lower} " in nome_padded or \
           f"-{sigla_lower}-" in nome_padded or \
           f" {sigla_lower}(" in nome_padded or \
           f"_{sigla_lower}_" in nome_padded or \
           f" {sigla_lower}." in nome_padded:
            segmentos_encontrados.append(sigla)
    
    # Usa set para remover duplicatas caso mais de um padrão corresponda
    segmentos_unicos = list(set(segmentos_encontrados))

    if not segmentos_unicos:
        return ["Geral"]
    return segmentos_unicos

def chunk_texto(texto, tamanho_chunk=2000, sobreposicao=200):
    """Divide o texto em chunks (pedaços) com sobreposição."""
    if not texto: return []
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho_chunk
        chunks.append(texto[inicio:fim])
        inicio += tamanho_chunk - sobreposicao
    return chunks

def processar_pasta_comunicados():
    """Orquestra o processo de ler, chunkear, gerar embeddings e salvar."""
    base_conhecimento_vetorial = []
    print(f"Iniciando processamento da pasta '{PASTA_COMUNICADOS}'...")

    if not os.path.isdir(PASTA_COMUNICADOS):
        print(f"Erro: Pasta '{PASTA_COMUNICADOS}' não encontrada.")
        return

    arquivos_pdf = [f for f in os.listdir(PASTA_COMUNICADOS) if f.lower().endswith(".pdf")]
    total_arquivos = len(arquivos_pdf)
    
    for i, nome_arquivo in enumerate(arquivos_pdf):
        print(f"Processando arquivo {i+1}/{total_arquivos}: {nome_arquivo}...")
        caminho_completo = os.path.join(PASTA_COMUNICADOS, nome_arquivo)
        
        texto_completo = extrair_texto_pdf(caminho_completo)
        if not texto_completo:
            continue

        segmentos = identificar_segmentos(nome_arquivo)
        print(f"  - Segmentos identificados: {segmentos}") # Log para depuração
        chunks_de_texto = chunk_texto(texto_completo)
        
        documento_processado = {
            "arquivo": nome_arquivo,
            "segmentos": segmentos,
            "chunks": []
        }
        
        for j, chunk in enumerate(chunks_de_texto):
            try:
                # API pode ter limite de requisições por minuto, adicionamos um delay
                time.sleep(1) 
                print(f"    - Gerando embedding para o chunk {j+1}/{len(chunks_de_texto)}...")
                embedding_result = genai.embed_content(model=MODELO_EMBEDDING, content=chunk)
                
                documento_processado["chunks"].append({
                    "texto_chunk": chunk,
                    "vetor": embedding_result['embedding']
                })
            except Exception as e:
                print(f"    !! Erro ao gerar embedding para o chunk: {e}. Pulando este chunk.")

        base_conhecimento_vetorial.append(documento_processado)

    print("-" * 20)
    print("Salvando a base de conhecimento vetorial...")
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(base_conhecimento_vetorial, f, ensure_ascii=False, indent=2)
    
    print("Processamento concluído com sucesso!")
    print(f"A base de conhecimento vetorial foi salva em '{ARQUIVO_SAIDA_JSON}'.")

if __name__ == "__main__":
    processar_pasta_comunicados()