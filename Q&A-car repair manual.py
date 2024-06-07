import fitz  # PyMuPDF
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# OpenAI API key
openai.api_key = 'sk-proj-CyKOo4xpw7GrXR5TTZ3PT3BlbkFJMWanpEG728tIFj9J5RXd'

# File path
pdf_path = "F:\Company_Data\Script_python\case_study\Car Repair Guide.pdf"

def extract_text_from_pdf(pdf_path):
#Extracts text from a PDF file.
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

#Chunking - chain of thought criteria
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Embedding
def embed_texts(texts, model):
    return model.encode(texts)

# Contextual similarity
def find_relevant_chunks(chunks, query, model, top_k=3):
    query_embedding = model.encode([query])
    chunk_embeddings = embed_texts(chunks, model)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    relevant_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in relevant_indices]

def generate_answer(query, context):
#Generates an answer to the query using OpenAI API with the given context.
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Chunk the text
    chunks = chunk_text(text)

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Questions
    questions = [
        "What precautions need to be taken in terms of clothing and hair?",
        "When and how to check engine oil level?",
        "Below what temperature do we need antifreeze washer fluid?"
    ]

    # Generate answers
    for question in questions:
        relevant_chunks = find_relevant_chunks(chunks, question, model)
        context = " ".join(relevant_chunks)
        answer = generate_answer(question, context)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()























