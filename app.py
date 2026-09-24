import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# ------------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="El Fermentador de PDFs",
    page_icon="🥛",
    layout="centered",
)

# ------------------------------------------------------------------
# Estética kéfir: crema, leche, kraft, madera y yute
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Nunito:wght@400;600&display=swap');

    :root {
        --leche: #FFFDF7;
        --crema: #F6EEDC;
        --kraft: #D9B98C;
        --yute: #B8925A;
        --madera: #8A5A2B;
        --tinta: #3B3A6B;
        --cafe: #4A3522;
    }

    /* Fondo general: mesa de madera clara */
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(255,253,247,0.9) 0%, rgba(246,238,220,0.95) 45%, rgba(233,215,183,1) 100%);
        color: var(--cafe);
        font-family: 'Nunito', sans-serif;
    }

    /* Títulos */
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: var(--cafe) !important;
        letter-spacing: -0.5px;
    }

    .titulo-kefir {
        font-family: 'Caveat', cursive;
        font-size: 3.4rem;
        color: var(--tinta);
        text-align: center;
        line-height: 1;
        margin-bottom: 0.2rem;
    }

    .subtitulo-kefir {
        font-family: 'Fraunces', serif;
        text-align: center;
        font-size: 1.05rem;
        color: var(--madera);
        margin-bottom: 1.2rem;
    }

    /* Etiqueta kraft, como la del frasco */
    .etiqueta {
        background: var(--kraft);
        color: var(--tinta);
        font-family: 'Caveat', cursive;
        font-size: 1.6rem;
        display: inline-block;
        padding: 0.2rem 1.2rem 0.2rem 1.6rem;
        border-radius: 6px 14px 14px 6px;
        position: relative;
        box-shadow: 2px 3px 0 rgba(138,90,43,0.25);
        margin: 0.6rem 0 0.4rem 0;
    }
    .etiqueta::before {
        content: "";
        position: absolute;
        left: 0.5rem; top: 50%;
        width: 8px; height: 8px;
        margin-top: -4px;
        border-radius: 50%;
        background: var(--leche);
        box-shadow: inset 1px 1px 2px rgba(0,0,0,0.25);
    }

    /* Tarjeta tipo frasco de vidrio */
    .frasco {
        background: rgba(255,253,247,0.85);
        border: 2px solid rgba(184,146,90,0.35);
        border-radius: 22px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 8px 24px rgba(138,90,43,0.12);
        margin-bottom: 1rem;
    }

    /* Imagen con marco de yute */
    [data-testid="stImage"] img {
        border-radius: 20px;
        border: 6px solid var(--crema);
        box-shadow: 0 10px 28px rgba(138,90,43,0.25);
    }

    /* Barra lateral: saco de yute */
    [data-testid="stSidebar"] {
        background:
            repeating-linear-gradient(45deg, rgba(184,146,90,0.10) 0 3px, transparent 3px 7px),
            repeating-linear-gradient(-45deg, rgba(184,146,90,0.10) 0 3px, transparent 3px 7px),
            #EAD8B5;
        border-right: 3px dashed rgba(138,90,43,0.35);
    }
    [data-testid="stSidebar"] * { color: var(--cafe) !important; }

    /* Campos de texto: leche espesa */
    .stTextInput input, .stTextArea textarea {
        background: var(--leche) !important;
        border: 2px solid var(--kraft) !important;
        border-radius: 14px !important;
        color: var(--cafe) !important;
        font-family: 'Nunito', sans-serif !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--madera) !important;
        box-shadow: 0 0 0 3px rgba(217,185,140,0.45) !important;
    }

    /* Cargador de archivos: cuchara de madera */
    [data-testid="stFileUploader"] section {
        background: var(--leche);
        border: 2px dashed var(--yute);
        border-radius: 18px;
    }

    /* Botones */
    .stButton button {
        background: var(--madera);
        color: var(--leche);
        border: none;
        border-radius: 999px;
        font-family: 'Fraunces', serif;
        padding: 0.5rem 1.4rem;
    }
    .stButton button:hover { background: var(--cafe); color: var(--leche); }

    /* Respuesta: servida en taza */
    .respuesta {
        background: var(--leche);
        border-left: 6px solid var(--kraft);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        font-size: 1.02rem;
        line-height: 1.6;
        box-shadow: 0 6px 18px rgba(138,90,43,0.12);
    }

    /* Burbujas de fermentación decorativas */
    .burbujas {
        text-align: center;
        font-size: 1.3rem;
        letter-spacing: 0.6rem;
        opacity: 0.55;
        margin: 0.4rem 0 1rem 0;
    }

    .pie {
        text-align: center;
        font-family: 'Caveat', cursive;
        font-size: 1.3rem;
        color: var(--madera);
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.markdown('<div class="titulo-kefir">El Fermentador de PDFs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo-kefir">Generación Aumentada por Recuperación (RAG), '
    'cultivada con paciencia como un buen kéfir</div>',
    unsafe_allow_html=True,
)

try:
    image = Image.open("kefiii.jpg")
    st.image(image, width=620)
except Exception as e:
    st.warning(f"No encontré la foto del frasco (kefiii.jpg): {e}")

st.markdown('<div class="burbujas">○ ◌ ○ ◯ ○ ◌ ○</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Barra lateral: la receta
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🥛 La receta")
    st.markdown(
        """
        Este agente trabaja como un frasco de kéfir:

        **1. La leche** 🍶
        Pones tu clave de OpenAI.

        **2. Los nódulos** 🫧
        Subes tu PDF. Lo partimos en fragmentos pequeños, como granitos de kéfir.

        **3. La fermentación** ⏳
        Cada fragmento se convierte en vectores y reposa en el frasco (FAISS).

        **4. A la cuchara** 🥄
        Haces una pregunta y servimos solo los fragmentos que la responden.
        """
    )
    st.markdown("---")
    st.caption(f"Cultivado con Python {platform.python_version()}")
    st.caption("Consejo de la casa: preguntas concretas, respuestas más cremosas.")

# ------------------------------------------------------------------
# Paso 1: la leche (clave de API)
# ------------------------------------------------------------------
st.markdown('<span class="etiqueta">1 · La leche</span>', unsafe_allow_html=True)
ke = st.text_input(
    "Vierte tu clave de OpenAI en el frasco",
    type="password",
    placeholder="sk-...",
)
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.info("🍶 Sin leche no hay kéfir: ingresa tu clave de OpenAI para empezar.")

# ------------------------------------------------------------------
# Paso 2: los nódulos (PDF)
# ------------------------------------------------------------------
st.markdown('<span class="etiqueta">2 · Los nódulos</span>', unsafe_allow_html=True)
pdf = st.file_uploader("Agrega tu PDF al cultivo", type="pdf")


@st.cache_resource(show_spinner=False)
def fermentar(texto: str, _clave: str):
    """Parte el texto en nódulos y los deja reposar en FAISS.
    Se guarda en caché para no volver a fermentar el mismo PDF en cada pregunta."""
    separador = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    nodulos = separador.split_text(texto)
    embeddings = OpenAIEmbeddings()
    frasco = FAISS.from_texts(nodulos, embeddings)
    return frasco, len(nodulos)


# ------------------------------------------------------------------
# Paso 3 y 4: fermentar y servir
# ------------------------------------------------------------------
if pdf is not None and ke:
    try:
        with st.spinner("🫧 Extrayendo la leche del documento..."):
            lector = PdfReader(pdf)
            texto = ""
            for pagina in lector.pages:
                texto += pagina.extract_text() or ""

        if not texto.strip():
            st.warning(
                "Este PDF parece venir vacío o escaneado como imagen. "
                "No hay leche que fermentar."
            )
            st.stop()

        with st.spinner("⏳ Fermentando... los nódulos están trabajando"):
            frasco, n_nodulos = fermentar(texto, ke)

        col1, col2 = st.columns(2)
        col1.metric("🍶 Leche extraída", f"{len(texto):,} caracteres".replace(",", "."))
        col2.metric("🫧 Nódulos cultivados", n_nodulos)

        st.markdown('<span class="etiqueta">3 · A la cuchara</span>', unsafe_allow_html=True)
        pregunta = st.text_area(
            "¿Qué quieres saber de tu documento?",
            placeholder="Escribe tu pregunta aquí y la servimos fresquita...",
        )

        if pregunta:
            with st.spinner("🥄 Sirviendo tu respuesta..."):
                docs = frasco.similarity_search(pregunta)
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
                cadena = load_qa_chain(llm, chain_type="stuff")
                respuesta = cadena.run(input_documents=docs, question=pregunta)

            st.markdown("### 🥛 Tu respuesta, recién fermentada")
            st.markdown(f'<div class="respuesta">{respuesta}</div>', unsafe_allow_html=True)

            with st.expander("🫧 Ver los nódulos que usamos para responder"):
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Nódulo {i}**")
                    st.caption(d.page_content)

    except Exception as e:
        st.error(f"Algo se cortó en la fermentación: {e}")
        import traceback
        st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("🍶 Ya tienes los nódulos, pero falta la leche: ingresa tu clave de OpenAI.")
else:
    st.info("🥄 Carga un PDF para empezar a fermentar.")

st.markdown('<div class="pie">hecho con paciencia, bacterias buenas y un poco de IA</div>',
            unsafe_allow_html=True)
