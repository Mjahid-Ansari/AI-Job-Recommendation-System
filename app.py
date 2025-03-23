import streamlit as st
import spacy
import numpy as np
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
from pyngrok import ngrok
import os

# Allow Streamlit to run in Jupyter Notebook
nest_asyncio.apply()

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return ' '.join(skills)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# IT Job Roles (Expanded to 15)
jobs = [
    "Data Scientist with expertise in Machine Learning, Python, and Deep Learning.",
    "Full Stack Developer skilled in JavaScript, React.js, Node.js, and MongoDB.",
    "AWS Cloud Engineer with hands-on experience in Terraform, Kubernetes, and Lambda.",
    "Cybersecurity Analyst experienced in network security, penetration testing, and SIEM tools.",
    "DevOps Engineer proficient in Docker, Kubernetes, Jenkins, and CI/CD automation.",
    "AI Engineer with strong knowledge of NLP, TensorFlow, and Generative AI.",
    "Database Administrator experienced in MySQL, PostgreSQL, and MongoDB.",
    "Software Engineer with expertise in Java, Spring Boot, and Microservices.",
    "Backend Developer skilled in Node.js, Express.js, and GraphQL.",
    "Frontend Developer experienced in React.js, Next.js, and Tailwind CSS.",
    "Mobile App Developer specializing in React Native, Flutter, and Swift.",
    "Blockchain Developer with experience in Solidity, Ethereum, and smart contracts.",
    "Data Engineer skilled in Spark, Hadoop, and ETL pipelines.",
    "IT Support Specialist with experience in troubleshooting, networking, and system administration.",
    "Product Manager with a strong background in Agile methodologies and business analytics."
]

# Preprocess Job Descriptions
processed_jobs = [extract_skills(job) for job in jobs]
tfidf = TfidfVectorizer()
job_vectors = tfidf.fit_transform(processed_jobs)

# Streamlit App Code
app_code = f"""
import streamlit as st
import spacy
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Function to extract skills
def extract_skills(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return ' '.join(skills)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# IT Job Roles
jobs = {jobs}

# Preprocess Job Descriptions
processed_jobs = [extract_skills(job) for job in jobs]
tfidf = TfidfVectorizer()
job_vectors = tfidf.fit_transform(processed_jobs)

st.title("💼 Resume-Based Job Recommendation System")

uploaded_file = st.file_uploader("📂 Upload your Resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
    elif file_type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)

    processed_resume = extract_skills(resume_text)
    resume_vector = tfidf.transform([processed_resume])
    
    scores = cosine_similarity(resume_vector, job_vectors)
    best_match_index = np.argmax(scores)
    
    st.success(f"✅ Recommended Job: {{jobs[best_match_index]}}")
"""

# Save the Streamlit app code to a file (Fix Unicode Error)
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

# Start the Streamlit app in the background
os.system("streamlit run app.py &")

# Open a public URL using ngrok (if needed)
public_url = ngrok.connect(8501)
print("Streamlit app is running at:", public_url)
