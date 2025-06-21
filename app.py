import streamlit as st
import os
import json
import numpy as np
import google.generativeai as genai
from PIL import Image # <-- 1. IMPORTAMOS A BIBLIOTECA DE IMAGEM

# --- CARREGAR LOGO ---
# 2. TENTAMOS ABRIR O ARQUIVO DO LOGO
try:
    logo = Image.open("logo.png")
except FileNotFoundError:
    # Caso o arquivo não seja encontrado, usamos um emoji padrão
    logo = "🤖"

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="CarboBot | Colégio Carbonell",
    page_icon="🤖",
    layout="centered"
)

# --- CONFIGURAÇÃO DA API DO GEMINI ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    st.error("🚨 Chave da API do Gemini não encontrada. Por favor, configure o arquivo secrets.toml.")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def carregar_base_conhecimento():
    """Carrega os dados do arquivo JSON da base de conhecimento vetorial."""
    try:
        with open("base_conhecimento.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None # Será tratado na tela de chat
    except json.JSONDecodeError:
        return "erro_json" # Sinaliza erro de formato

def encontrar_chunks_relevantes(pergunta_usuario, base_conhecimento, sigla_segmento, top_k=4):
    """Encontra os chunks mais relevantes para a pergunta usando busca vetorial com numpy."""
    try:
        # 1. Gerar o embedding da pergunta do usuário
        vetor_pergunta = genai.embed_content(
            model='models/text-embedding-004',
            content=pergunta_usuario
        )['embedding']
    except Exception as e:
        st.error(f"Não foi possível gerar o vetor para a sua pergunta. Erro: {e}")
        return []

    # 2. Filtrar os documentos pelo segmento selecionado
    docs_segmento = [
        doc for doc in base_conhecimento
        if sigla_segmento in doc["segmentos"] or "Geral" in doc["segmentos"]
    ]

    # 3. Calcular a similaridade (dot product) e coletar os chunks
    chunks_com_similaridade = []
    for doc in docs_segmento:
        for chunk in doc["chunks"]:
            # Garante que o vetor do chunk é um array numpy
            vetor_chunk = np.array(chunk["vetor"])
            # Calcula o produto escalar para medir a similaridade
            similaridade = np.dot(vetor_pergunta, vetor_chunk)
            
            chunks_com_similaridade.append({
                "texto": chunk["texto_chunk"],
                "fonte": doc["arquivo"],
                "similaridade": similaridade
            })
    
    # 4. Ordenar os chunks pela maior similaridade
    chunks_com_similaridade.sort(key=lambda x: x['similaridade'], reverse=True)
    
    # 5. Retornar os top_k chunks mais relevantes
    return chunks_com_similaridade[:top_k]

# --- GERENCIAMENTO DE ESTADO ---
if 'segmento_selecionado' not in st.session_state:
    st.session_state.segmento_selecionado = None
if 'mensagens' not in st.session_state:
    st.session_state.mensagens = []

# --- TELAS DA APLICAÇÃO ---
def tela_selecao_segmento():
    """Exibe a tela inicial para seleção de segmento."""
    st.image("logo.png", width=180)
    st.title(" Bem-vindo ao CarboBot")
    st.write("Seu assistente virtual para os comunicados do Colégio Carbonell.")
    st.markdown("---")
    st.write("Para começar, por favor, **selecione o segmento** do aluno:")

    segmentos = {"EI": "Ensino Infantil", "AI": "Anos Iniciais", "AF": "Anos Finais", "EM": "Ensino Médio"}
    
    col1, col2 = st.columns(2)
    
    def criar_botao_segmento(coluna, sigla):
        with coluna:
            if st.button(segmentos[sigla], use_container_width=True):
                st.session_state.segmento_selecionado = segmentos[sigla]
                st.session_state.sigla_segmento = sigla
                st.session_state.mensagens = [{"role": "assistant", "content": f"Olá! 👋 Como posso ajudar com os comunicados do {segmentos[sigla]}?"}]
                st.rerun()

    criar_botao_segmento(col1, "EI")
    criar_botao_segmento(col1, "AF")
    criar_botao_segmento(col2, "AI")
    criar_botao_segmento(col2, "EM")

def tela_chat():
    """Exibe a interface principal do chat, com busca vetorial e botões de download."""
    st.title(f"Chat - {st.session_state.segmento_selecionado}")
    
    base_conhecimento = carregar_base_conhecimento()
    if not base_conhecimento or base_conhecimento == "erro_json":
        st.error("A base de conhecimento não foi carregada corretamente. Execute `processar_pdfs.py` e reinicie o app.")
        if st.button("← Voltar"): st.session_state.segmento_selecionado = None; st.rerun()
        st.stop()

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Lógica para re-exibir botões de download em mensagens antigas da IA
            if msg.get("fontes"):
                for nome_arquivo_fonte in msg["fontes"]:
                    caminho_pdf = os.path.join("comunicados", nome_arquivo_fonte)
                    if os.path.exists(caminho_pdf):
                        with open(caminho_pdf, "rb") as f:
                            st.download_button(
                                label=f"📄 Baixar: {nome_arquivo_fonte}",
                                data=f.read(),
                                file_name=nome_arquivo_fonte,
                                mime="application/pdf",
                                key=f"download_{nome_arquivo_fonte}_{msg['content']}" 
                            )

    if prompt := st.chat_input("Qual a sua dúvida sobre os comunicados?"):
        st.session_state.mensagens.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando a informação mais relevante... 🧠"):
                chunks_encontrados = encontrar_chunks_relevantes(prompt, base_conhecimento, st.session_state.sigla_segmento)
                
                if not chunks_encontrados:
                    resposta_ia = "Nos comunicados disponíveis, não encontrei uma resposta para a sua pergunta. Tente perguntar de outra forma."
                    fontes = []
                    st.markdown(resposta_ia)
                else:
                    contexto_formatado = "\n\n".join(
                        [f"Fonte: {chunk['fonte']}\nTrecho Relevante: {chunk['texto']}" for chunk in chunks_encontrados]
                    )
                    fontes = list(set(chunk['fonte'] for chunk in chunks_encontrados))
                    
                    prompt_para_ia = f"""
                    Você é um assistente virtual amigável e prestativo da escola. Sua tarefa é responder perguntas dos pais com base nos trechos de comunicados oficiais fornecidos abaixo. Use um tom cordial e ajude da melhor forma possível.
                    **Contexto dos Trechos Relevantes:**
                    {contexto_formatado}
                    **Regras Importantes:**
                    1. Sempre baseie sua resposta principal nas informações do contexto. NUNCA invente datas, valores ou detalhes.
                    2. Formule uma resposta clara e amigável. Você pode começar com uma saudação como "Olá!" ou "Com certeza!".
                    3. Se a resposta exata não estiver no contexto, você pode dizer algo como: "Nos comunicados que consultei, não encontrei o detalhe exato sobre sua pergunta, mas a informação mais próxima que achei foi sobre [mencione o assunto do chunk encontrado].Caso não tenha encontrado oque procura, entre em contato com a secretária do Carbonell."
                    4. Ao final da sua resposta, cite o(s) nome(s) do(s) arquivo(s) fonte, assim: "Fonte(s): {', '.join(fontes)}".
                    **Pergunta do Usuário:**
                    {prompt}
                    """
                    with st.spinner("Formulando a melhor resposta... ✍️"):
                        try:
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            generation_config = genai.types.GenerationConfig(temperature=0.2)
                            response = model.generate_content(prompt_para_ia, generation_config=generation_config)
                            resposta_ia = response.text
                        except Exception as e:
                            resposta_ia = f"Desculpe, ocorreu um erro ao contatar a IA. Detalhes: {e}"
                    
                    st.markdown(resposta_ia)
                    
                    # --- NOVA FUNCIONALIDADE: BOTÕES DE DOWNLOAD ---
                    st.markdown("---") # Adiciona uma linha divisória
                    for nome_arquivo_fonte in fontes:
                        caminho_pdf = os.path.join("comunicados", nome_arquivo_fonte)
                        if os.path.exists(caminho_pdf):
                            with open(caminho_pdf, "rb") as f:
                                st.download_button(
                                    label=f"📄 Baixar: {nome_arquivo_fonte}",
                                    data=f.read(),
                                    file_name=nome_arquivo_fonte,
                                    mime="application/pdf",
                                    key=f"download_{nome_arquivo_fonte}_{prompt}" # Chave única para o botão
                                )
                        else:
                            st.warning(f"Arquivo da fonte '{nome_arquivo_fonte}' não foi encontrado.")
            
            # Adiciona a resposta e as fontes ao histórico para re-exibição
            st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia, "fontes": fontes})

    if st.button("← Voltar e selecionar outro segmento"):
        st.session_state.segmento_selecionado = None; st.session_state.mensagens = []; st.rerun()


# --- LÓGICA PRINCIPAL ---
if st.session_state.segmento_selecionado is None:
    tela_selecao_segmento()
else:
    tela_chat()