import streamlit as st
import fitz  # PyMuPDF
import docx
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load Summarization and QA models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    return summarizer, tokenizer, model

summarizer, tokenizer, qa_model = load_models()

# Functions to Read Files
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def summarize_text(text):
    max_chunk = 1000
    text = text.replace('\n', ' ')
    sentences = text.split('. ')
    current_chunk = ''
    chunks = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '. '
    chunks.append(current_chunk)

    result = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        result.append(summary)

    return ' '.join(result)

def answer_question(context, question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    return answer

# Streamlit UI
st.title("Real-Time Deep Learning Legal Summarizer + Chatbot")
st.markdown("Upload your legal document, get the summary, and ask questions!")

uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith('.pdf'):
        document_text = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        document_text = read_docx(uploaded_file)
    else:
        document_text = uploaded_file.read().decode('utf-8')

    with st.expander("See extracted document text"):
        st.write(document_text)

    if st.button("Summarize Document"):
        with st.spinner("Summarizing... please wait"):
            summary = summarize_text(document_text)

        st.subheader("Summary:")
        st.success(summary)

    st.subheader("Chatbot - Ask your document!")
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        with st.spinner("Thinking..."):
            answer = answer_question(document_text, user_question)
        st.info(f"Answer: {answer}")
