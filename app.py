import os
import platform

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Titulo y presentacion
st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.write("Versión de Python:", platform.python_version())

# Imagen (opcional)
try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Barra lateral
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Clave de OpenAI (cada usuario pone la suya)
ke = st.text_input("Ingresa tu Clave de OpenAI", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke

# Carga del PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")


@st.cache_resource(show_spinner="Procesando el documento...")
def construir_base(pdf_bytes: bytes, api_key: str):
    """Extrae el texto, lo divide en fragmentos y crea la base vectorial.
    Se guarda en cache para no volver a generar embeddings en cada interaccion."""
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    base = FAISS.from_texts(chunks, embeddings)
    return base, len(text), len(chunks)


if pdf is not None and ke:
    try:
        knowledge_base, n_chars, n_chunks = construir_base(pdf.getvalue(), ke)

        if n_chars == 0:
            st.error("No se pudo extraer texto del PDF. Puede ser un PDF escaneado (solo imágenes).")
            st.stop()

        st.info(f"Texto extraído: {n_chars} caracteres")
        st.success(f"Documento dividido en {n_chunks} fragmentos")

        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=4)
            contexto = "\n\n".join(d.page_content for d in docs)

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=ke)

            prompt = (
                "Responde la pregunta usando solo el contexto del documento. "
                "Si la respuesta no está en el contexto, dilo claramente.\n\n"
                f"Contexto:\n{contexto}\n\n"
                f"Pregunta: {user_question}\n\n"
                "Respuesta:"
            )

            with st.spinner("Pensando..."):
                response = llm.invoke(prompt)

            st.markdown("### Respuesta:")
            st.markdown(response.content)

    except Exception as e:
        import traceback
        st.error(f"Error al procesar el PDF: {e}")
        st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
elif not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
