 AI-Powered Resume-Based Job Recommendation System

This project leverages Natural Language Processing (NLP) and Machine Learning to analyze resumes and recommend the most suitable IT job roles based on skill matching. It uses TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to compare extracted skills from resumes with predefined job descriptions.
🔹 Features

✅ Upload resumes in PDF or TXT format
✅ AI-driven skill extraction & job matching
✅ Supports 15+ IT job roles
✅ Built using Streamlit, Scikit-Learn, and SpaCy
✅ Simple and user-friendly web interface
📂 Files in This Repository

    app.py – Main Streamlit application

    requirements.txt – List of dependencies

    README.md – Project documentation

🏗 Installation Guide

Follow these steps to set up and run the project locally:

    Clone the Repository

git clone https://github.com/Mjahid-Ansari/AI-Job-Recommendation-System.git
cd resume-job-recommender

Install Dependencies

pip install -r requirements.txt

Run the Application

    streamlit run app.py

📝 How It Works

    Upload your resume in PDF or TXT format.

    The system extracts skills from the resume using NLP.

    The extracted skills are compared against job descriptions using TF-IDF & Cosine Similarity.

    The best-matched IT job role is recommended based on similarity scores.

📌 Dependencies (requirements.txt)

streamlit
spacy
numpy
pandas
PyPDF2
scikit-learn

🚀 Get Started

Experience AI-powered job recommendations tailored to your skills! Fork the repository, contribute, and help improve this project!
