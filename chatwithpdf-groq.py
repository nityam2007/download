import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import hashlib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="ChatWithPDF", page_icon="📄", layout="wide")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# List of available models
MODELS = [
    "llama3-8b-8192",
    "gemma-7b-it",
    "gemma2-9b-it",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "mixtral-8x7b-32768"
]

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        try:
            text += page.extract_text() + "\n"
        except Exception as e:
            st.warning(f"Skipped a page due to: {str(e)}")
    return text

def split_into_chunks(text, chunk_size=1500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_or_create_chunks(file):
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)

    cache_file = f"cache/{file_hash}_chunks.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    text = process_pdf(file)
    chunks = split_into_chunks(text)

    os.makedirs('cache', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(chunks, f)

    return chunks

def find_most_relevant_chunks(query, chunks, top_k=2):
    vectorizer = TfidfVectorizer().fit(chunks + [query])
    chunk_vectors = vectorizer.transform(chunks)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def main():
    st.title("ChatWithPDF")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'model' not in st.session_state:
        st.session_state.model = MODELS[0]
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []

    st.sidebar.header("Upload PDF")
    pdf_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file:
        with st.spinner("Processing PDF..."):
            st.session_state.chunks = get_or_create_chunks(pdf_file)
        st.sidebar.success("PDF processed successfully!")

    selected_model = st.selectbox("Select Model", MODELS, index=MODELS.index(st.session_state.model))
    if selected_model != st.session_state.model:
        st.session_state.model = selected_model
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                relevant_chunks = find_most_relevant_chunks(prompt, st.session_state.chunks) if st.session_state.chunks else []
                context = "\n\n".join(relevant_chunks)
                enhanced_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"

                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for answering questions about the given PDF content."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    model=st.session_state.model,
                    max_tokens=2048,
                    temperature=0.7
                )
                full_response = chat_completion.choices[0].message.content

                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
