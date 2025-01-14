import os
import re
import io
import json
import base64
import zipfile

import streamlit as st
from PIL import Image
import pdf2image
import docx
import pandas as pd
import pytesseract
import google.generativeai as genai
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy
from spacy.matcher import PhraseMatcher
import nltk
import openai
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

class ATS_System:
    def __init__(self):
        st.set_page_config(layout="wide")
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            st.error("API key not loaded. Check your .env file for the OPENAI_API_KEY variable.")
        else:
            genai.configure(api_key=self.api_key)

        # Custom CSS for Styling (This part is kept from the original code for styling)
        st.markdown(
            """
            """,
            unsafe_allow_html=True,
        )
        # Header
        st.markdown(
            """
            📄 ATS Tracking System 🔍
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ↪️ Analyze Your Resume for JD Compatibility and Similarity
            
            Upload your Job Description and Resume, and get the Resume Analysis & Cosine Similarity score as per Job Description."
            """,
            unsafe_allow_html=True,
        )
        self.RESUME_FOLDER = "user_resumes"
        os.makedirs(self.RESUME_FOLDER, exist_ok=True)
        self.RESUME_FOLDER_1 = "ATS_SCORE_ABOVE_55"
        os.makedirs(self.RESUME_FOLDER_1, exist_ok=True)
        self.RESUME_FOLDER_2 = 'ATS_SCORE_BELOW_55'
        os.makedirs(self.RESUME_FOLDER_2, exist_ok=True)
        # OCR Configuration
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  #set tesseract path here
        
        self.IT_SKILLS = [
            "python", "java", "c++", "problem-solving", "algorithm design", "debugging", "git", "unit testing", "html", "css",
            "javascript", "react", "node.js", "express.js", "restful apis", "mongodb", "sql", "agile methodology", "ruby",
            "android", "ios", "react native", "swift", "kotlin", "flutter", "app design", "mobile ui/ux", "unity",
            "unreal engine", "game physics", "3d modeling", "multiplayer design", "ai for games", "docker", "kubernetes",
            "aws", "ci/cd", "infrastructure as code", "ansible", "jenkins", "cloud services", "microcontrollers", "rtos",
            "circuit design", "hardware-software integration", "sensors", "iot", "python", "r", "machine learning",
            "data analysis", "sql", "pandas", "numpy", "data visualization", "deep learning", "statistics", "excel",
            "tableau", "power bi", "data cleaning", "statistical analysis", "report generation", "tensorflow", "pytorch",
            "keras", "data preprocessing", "model evaluation", "neural networks", "regression", "natural language processing",
            "reinforcement learning", "research papers", "algorithms", "data warehousing", "etl", "data modeling",
            "big data technologies", "hadoop", "spark", "etl pipelines", "threat analysis", "risk assessment",
            "penetration testing", "network security", "firewalls", "siem", "encryption", "ethical hacking", "kali linux",
            "metasploit", "nmap", "vulnerability assessment", "exploit writing", "web application security",
            "network security", "cryptography", "risk management", "security governance", "compliance", "incident management",
            "forensics", "malware analysis", "security tools", "soc", "aws", "azure", "google cloud",
            "cloud infrastructure design", "cloud security", "automation", "networking", "terraform", "ci/cd",
            "cloud platforms", "virtualization", "servers", "storage solutions", "tcp/ip", "routing and switching",
            "vpns", "wireless networks", "network protocols", "storage systems", "nas", "san", "data backup",
            "disaster recovery", "storage", "project management", "product management", "scrum master", "it manager",
            "chief technology officer (cto)", "agile", "stakeholder management", "risk management", "budgeting",
            "team leadership", "strategic planning", "product lifecycle", "software testing", "qa engineering",
            "automation testing", "performance testing", "test management", "selenium", "junit", "cucumber",
            "bug tracking", "test plans", "load testing", "ui testing", "unit testing", "regression testing",
            "manual testing", "security testing", "network administration", "help desk", "it support", "voip",
            "networking", "linux", "windows", "mac os", "remote desktop", "troubleshooting", "hardware setup", "dns",
            "dhcp", "vpn", "firewalls", "tcp/ip", "remote access", "cloud support", "ip addressing", "routing",
            "switching", "it consulting", "business analysis", "systems analysis", "stakeholder management",
            "project management", "enterprise architecture", "it strategy", "change management", "risk management",
            "consulting methodology", "requirements gathering", "client communication", "business process improvement",
            "systems integration"
        ]

        self.input_prompt1 = """
            You are a Technical HR Manager. Evaluate the skills from the resume against the job description:
            - Match the technical skills from the job description with the resume and list only the matching skills in 4 lines.
            - Job Description Experience in bullet points [mention years of experience required as per JD, e.g., 5 years]
            - Candidate Resume Experience in bullet points [mention candidate's experience in years, e.g., 3 years] (Extracted from resume if available)
            - List only missing technical skills in bullet points in 4 lines , be concise.
            - Provide a suggestion Whether candidate should apply for job based on JD (e.g-You can apply for this job 😊) in 10 words.
        """

        self.input_prompt3 = """
            You are an ATS expert. Evaluate the resume’s keyword match with the job description.
            - Mention the ATS Score:- percentage to user in big title (Eg-ATS Score 85%)
            - Provide for overall keyword matching skills percentage (e.g-75%) in bullet points.
            - List missing keywords as per job description only concisely in bullet points.
            Keep the response concise, around 40 words.
        """

        self.input_prompt4 = """
            You are an ATS expert. Evaluate the resume’s keyword match with the job description.
            - Mention the ATS Score as per mention JD & Resume matching skills & experience:- percentage to user in big title (Eg-ATS Score 85%)
            - If ATS SCore is greater than 60% then provide feedback to user (eg.-Recruiter might call you soon,etc) as per your knowledge in medium heading.
            - If ATS SCore less than 60 then provide feedback to candidate improvement in resume(eg.-Get some experience,Learn AI Skills,etc) as per your knowledge.
            - Provide for overall keyword matching skills percentage (e.g-75%) in bullet points.
            - Give siggestion to candidate based on ATS SCore between JD & Resume matching skills,experience in 20 words in bullet point.
            - List missing keywords as per job description only concisely in bullet points.
            Keep the response concise, around 40 words .
        """

        self.output_placeholder = st.empty()
        self.response_placeholder = st.empty()
        if 'preprocess_JD' not in st.session_state:
            st.session_state.preprocess_JD = None
        if 'preprocess_doc_content' not in st.session_state:
            st.session_state.preprocess_doc_content = None
        if 'doc_content' not in st.session_state:
            st.session_state.doc_content = None
        if 'job_description_preprocess' not in st.session_state:
            st.session_state.job_description_preprocess = None
        if 'preprocess_doc_content_4' not in st.session_state:
            st.session_state.preprocess_doc_content_4 = None
        
        self.jd_skills_normalized = None
        self.resume_skills_normalized = None
        self.matching_skills = None
        self.similarity_score_1 = 0
        self.similarity_score = 0


    def input_pdf_setup(self, uploaded_file):
        try:
            temp_path = os.path.join(self.RESUME_FOLDER, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            images = convert_from_path(temp_path)
            os.remove(temp_path)
            text = "".join([pytesseract.image_to_string(img) for img in images])
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

    def input_docx_setup(self, uploaded_file):
        try:
            doc = docx.Document(uploaded_file)
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
            return text
        except Exception as e:
            st.error(f"Error processing DOCX: {e}")
            return None
    
    def input_image_setup(self, uploaded_file):
        try:
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def preprocess_text(self, text):
        text = text.replace(" @", "@").replace("@ ", "@")
        new_punctuation = list(punctuation)
        new_punctuation.remove('@')
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in stop_words and (word not in new_punctuation)]
        new = ' '.join(filtered_tokens)
        return new.replace(" @ ", "@").replace('\'','')
    
    def save_uploaded_file(self, uploaded_file):
        try:
            os.makedirs(self.RESUME_FOLDER, exist_ok=True)
            file_path = os.path.join(self.RESUME_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            st.error(f"Error saving the file: {e}")
            return None
    
    def ATS_RESUME_ABOVE_55(self, uploaded_file):
        try:
            os.makedirs(self.RESUME_FOLDER_1, exist_ok=True)
            file_path = os.path.join(self.RESUME_FOLDER_1, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            st.error(f"Error saving the file: {e}")
            return None

    def ATS_RESUME_BELOW_55(self, uploaded_file):
        try:
            os.makedirs(self.RESUME_FOLDER_2, exist_ok=True)
            file_path = os.path.join(self.RESUME_FOLDER_2, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            st.error(f"Error saving the file: {e}")
            return None
    
    def extract_skills_from_text(self, text, skill_set):
        nlp = spacy.blank("en")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(skill) for skill in skill_set]
        matcher.add("SKILLS", patterns)
        doc = nlp(text)
        matches = matcher(doc)
        return {doc[start:end].text for _, start, end in matches}

    def calculate_similarity(self, jd_skills, matching_skills):
         jd_text = self.preprocess_text(', '.join(jd_skills))
         matching_text = self.preprocess_text(', '.join(matching_skills))
         vectorizer = TfidfVectorizer()
         vectors = vectorizer.fit_transform([jd_text, matching_text])
         similarity = cosine_similarity(vectors[0:1], vectors[1:2])
         return similarity
    
    def normalize_skills(self, skills):
        return {skill.strip().lower() for skill in skills}
    
    def give_suggestion(self, similarity_score):
        if similarity_score >= 0.80:
            suggestion = 'Good Match! You have Many of Required Skills.Apply with confidence'
            color = "#4CAF50"
            font = "Times New Roman, serif"
        elif similarity_score >= 0.60:
            suggestion = "Good match! You have many of the required skills. Consider applying with confidence."
            color = "#4CAF50"
            font = "Times New Roman, serif"
        elif similarity_score >= 0.50:
            suggestion = "Decent match, but there are areas for improvement. You may want to consider applying with additional learning in relevant areas.😊"
            color = "#4CAF50"
            font = "Times New Roman, serif"
        else:
            suggestion = "Your skills don't closely match the job requirements. Consider gaining experience in the areas where you're lacking to strengthen your application."
            color = "#F44336"
            font = "Times New Roman, serif"
        return suggestion, color, font
    
    def give_suggestion_submit4(self, similarity_score):
        if similarity_score * 100 >= 70:
            suggestion_ = (
                "🌟 Great match! Your skills align very well with the job requirements. "
                "High chances of getting shortlisted! Recruiters might call you soon. 😊"
            )
            color_ = "#4CAF50"
            font_ = "Times New Roman, serif"
        elif similarity_score * 100 >= 50:
            suggestion_ = (
                "✅ Good match! You have many of the required skills. "
                "There is a fair chance of being shortlisted. Consider applying confidently! 😊"
            )
            color_ = "#4CAF50"
            font_ = "Times New Roman, serif"
        elif similarity_score*100 <50:
            suggestion_ = (
                "🟠 Decent match, but there are areas for improvement. "
                "Low chances of getting shortlisted unless you improve on the relevant skills. "
                "Consider gaining additional experience before applying. 😊"
            )
            color_ ="#F44336"
            font_ = "Times New Roman, serif"
        return suggestion_,color_,font_

    def display_skills_and_similarity_submit4(self, jd_text, resume_text):
        jd_skills = self.extract_skills_from_text(jd_text, self.IT_SKILLS)
        resume_skills = self.extract_skills_from_text(resume_text, self.IT_SKILLS)
        self.jd_skills_normalized = self.normalize_skills(jd_skills)
        self.resume_skills_normalized = self.normalize_skills(resume_skills)
        self.matching_skills = self.jd_skills_normalized.intersection(self.resume_skills_normalized)
        self.similarity_score_1 = 0
        if not self.matching_skills:
            st.markdown(f"No matching skills found between JD and Resume.", unsafe_allow_html=True)
        else:
            self.similarity_score_1 = self.calculate_similarity(self.jd_skills_normalized, self.matching_skills)
            suggestion_, color_, font_ = self.give_suggestion_submit4(self.similarity_score_1)
            st.markdown(f"{suggestion_}", unsafe_allow_html=True)
        data = {
            "JD Skills": [', '.join(self.jd_skills_normalized)],
            "Resume Skills": [', '.join(self.resume_skills_normalized)],
            "Matching Skills between JD & Resume": [', '.join(self.matching_skills) if self.matching_skills else 'None'],
            "Cosine Similarity": [f"{self.similarity_score_1 * 100:.2f}%" if self.similarity_score_1 else '0%']
            }
        df = pd.DataFrame(data)
        st.markdown("""
            """, unsafe_allow_html=True)
        st.markdown(f"Cosine Similarity between Job Description Skills and Matching Resume Skills as per JD:- {self.similarity_score_1 * 100:.2f}%", unsafe_allow_html=True)
        st.markdown(df.to_html(classes='dataframe', index=False), unsafe_allow_html=True)

    def display_skills_and_similarity(self, jd_text, resume_text):
        jd_skills = self.extract_skills_from_text(jd_text, self.IT_SKILLS)
        resume_skills = self.extract_skills_from_text(resume_text, self.IT_SKILLS)
        self.jd_skills_normalized = self.normalize_skills(jd_skills)
        self.resume_skills_normalized = self.normalize_skills(resume_skills)
        self.matching_skills = self.jd_skills_normalized.intersection(self.resume_skills_normalized)
        if not self.matching_skills:
            st.markdown(f"No matching skills found between JD and Resume.", unsafe_allow_html=True)
            self.similarity_score = 0
        else:
            self.similarity_score = self.calculate_similarity(self.jd_skills_normalized, self.matching_skills)
            suggestion, color, font = self.give_suggestion(self.similarity_score)
            st.markdown(f"{suggestion}", unsafe_allow_html=True)
        data = {
            "Job Description Skills": [', '.join(self.jd_skills_normalized)],
            "Resume Skills": [', '.join(self.resume_skills_normalized)],
            "Matching Skills between Job Description & Resume": [', '.join(self.matching_skills) if self.matching_skills else 'None'],
            "Cosine Similarity": [f"{self.similarity_score * 100:.2f}%" if 'similarity_score' in globals() else '0%']
            }
        df = pd.DataFrame(data)
        st.markdown("""
            """, unsafe_allow_html=True)
        st.markdown(f"Cosine Similarity between Job Description Skills and Matching Resume Skills as per JD:- {self.similarity_score * 100:.2f}%", unsafe_allow_html=True)
        st.markdown(df.to_html(classes='dataframe', index=False), unsafe_allow_html=True)

    def extract_jd_info(self, job_description):
        input_prompt_jd_extraction = """
            You are a Technical HR Analyst tasked with extracting key information from a job description. Please identify and extract the following details from the job description provided:
            - Job Title: Extract the title of the job being advertised.
            - Qualifications: List the required educational background and certifications.
            - Skills: List all technical and soft skills mentioned in the job description.
            - Experience: Identify the required years and type of experience.
            - Company: Identify the name of the company if mentioned in the job description.
            Format your response as a JSON object with the following keys: "job_title", "qualifications", "skills", "experience", and "company". If the company name cannot be determined, use the value "Not specified". If a field cannot be determined, use "Not specified" as value.
            Job Description:
        """
        response = self.get_openai_response(None,job_description, input_prompt_jd_extraction)
        if response:
            try:
                jd_data = json.loads(response)
                return jd_data
            except json.JSONDecodeError:
                st.error("Error: Gemini response is not in JSON format.")
                return None
        else:
            return None

    def get_openai_response(self, input_text, doc_content, prompt):
        try:
            if not input_text:
                st.error("Job description is missing. Please enter a job description.")
                return None
            if not doc_content:
                st.error("Resume content is missing. Please upload a resume.")
                return None
            if not prompt:
                st.error("Prompt is missing. Please provide a valid prompt.")
                return None
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": f"Job Description: {input_text}\n Resume:{doc_content}\n {prompt}"},
                ]
            )
            return response.choices.message['content']
        except Exception as e:
            st.error(f"Error during API call: {e}")
            return None
    
    def generate_cover_letter(self, job_description, resume_content):
        cover_letter_agent = Agent(
            role="Cover Letter Writer",
            goal="Craft a concise, professional, and tailored cover letter.",
            backstory=(
                "You specialize in writing compelling cover letters tailored "
                "to job descriptions and resumes. Your goal is to keep it concise, focusing on key strengths and qualifications."
            ),
            memory=False,
            tools=[]
        )
        cover_letter_task = Task(
            description=(
                "Generate a concise and impactful cover letter based on the provided resume and job description."
                "The cover letter should be professional, personalized, and highlight the candidate's skills strengths"
                "\nInputs:\n- Resume: {resume}\n- Job Description: {job_description}"
            ),
            expected_output="A structured, concise cover letter with an Introduction, Key Strengths, and Closing.",
            agent=cover_letter_agent
        )
        cover_letter_crew = Crew(
            agents=[cover_letter_agent],
            tasks=[cover_letter_task],
            process=Process.sequential
        )
        inputs = {
            'resume': resume_content,
            'job_description': job_description
        }
        try:
            result = cover_letter_crew.kickoff(inputs=inputs)
            task_output = cover_letter_task.output
            return task_output.raw
        except Exception as e:
             st.error(f"Error during cover letter generation: {str(e)}")
    
    def ask_openai(self, question, resume_text):
         prompt = f"""
            You are an expert resume analyzer. Answer the following question based only on the provided resume content. Do not include any external knowledge.
            Resume:
            {resume_text}
            Question:
            {question}
            Answer
        """
         try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert resume analyzer. Answer the following question based only on the provided resume content."},
                    {"role": "user", "content": f"Resume: {resume_text}"},
                    {"role": "user", "content": f"Question: {question}"}
                ],max_tokens = 200,
                temperature = 0.3
            )
            return response['choices']['message']['content']
         except Exception as e:
            st.error(f"Error with OpenAI API: {e}")
            return "Sorry, I couldn't process your query at the moment."
    
    def process_uploaded_file(self, uploaded_file):
        if uploaded_file.type == "application/pdf":
            doc_content = self.input_pdf_setup(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc_content = self.input_docx_setup(uploaded_file)
        elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            doc_content = self.input_image_setup(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or image file.")
            return None, None
        if doc_content:
            preprocess_doc_content = self.preprocess_text(doc_content)
            return doc_content, preprocess_doc_content
        else:
             return None, None
    
    def run(self):
        with st.container():
            input_jd = st.text_input("Job Description:", key="input", placeholder="Enter the job description here...")
            if input_jd:
                st.session_state.preprocess_JD = self.preprocess_text(input_jd)
            else:
                st.error('Please Mention Job Description! ')
            uploaded_file = st.file_uploader("Upload your resume:", type=["pdf", "docx", "jpg", "jpeg", "png"])
            if uploaded_file:
                pass
            else:
                st.error('Please Upload your Resume..')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                submit1 = st.button("Tell Me About the Resume")
            with col2:
                submit3 = st.button("Percentage Match")
            with col3:
                submit5 = st.button("Generate Cover Letter")
            with col4:
                submit6 = st.button('Chat with Your Resume')
            if submit1 or submit3:
                if uploaded_file:
                    self.save_uploaded_file(uploaded_file)
                    doc_content, preprocess_doc_content = self.process_uploaded_file(uploaded_file)
                    if preprocess_doc_content:
                        if input_jd:
                            if submit1:
                                self.display_skills_and_similarity(st.session_state.preprocess_JD, preprocess_doc_content)
                                response = self.get_openai_response(st.session_state.preprocess_JD, doc_content, self.input_prompt1)
                                st.write(response)
                            if submit3:
                                response = self.get_openai_response(st.session_state.preprocess_JD, doc_content, self.input_prompt3)
                                st.write(response)
                        else:
                            st.error('Please Mention Job Description')

            if submit5:
                 if uploaded_file:
                     self.save_uploaded_file(uploaded_file)
                     doc_content, preprocess_doc_content = self.process_uploaded_file(uploaded_file)
                     if preprocess_doc_content:
                         if input_jd:
                            ai_agents_cover_letter = self.generate_cover_letter(input_jd,doc_content)
                            st.subheader("Generated Cover Letter :")
                            st.write(ai_agents_cover_letter)
                         else:
                             st.error("Please mention Job Description..")
                 else:
                     st.error('Please Upload your resume')
            
            if submit6:
                if uploaded_file:
                    doc_content, preprocess_doc_content = self.process_uploaded_file(uploaded_file)
                    st.session_state.doc_content = doc_content
                if st.session_state.get('doc_content', None):
                   user_query = st.text_input("Ask a question about your resume")
                   send_button = st.button('Send Query..',
                   key='send_query_btn',
                   help='Click to send your query',
                   type='primary',
                    )
                   if user_query and send_button:
                        response = self.ask_openai(user_query, st.session_state.doc_content)
                        st.markdown("""
                <style>
                .response-container {
                    background-color: #1e3d59;
                    border-radius: 10px;
                    padding: 5px;
                    margin: 10px 0;
                    color: #ffffff;
                    border-left: 5px solid #17b978;
                }
                </style>
            """, unsafe_allow_html=True)
            try:
                if submit1:
                    text_output = preprocess_doc_content
                    NAME_REGEX = r"^[A-Za-z]+(?: [A-Za-z]+)+$|\b[a-zA-Z]+\s+[a-zA-Z]+\b"
                    PHONE_REGEX = r"(\+91[\s\-]?)?(\d{5})[\s\-]?(\d{2,8})|(\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}"
                    EMAIL_REGEX = r"\b[a-z0-9_.]{2,30}[@][gmail.com]{1,15}\b|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                    EXPERIENCE_REGEX = r'(\d+\.?\d*)\s+(years|yrs|year|yr)|(\d+(\.\d+)?)\s*(years|yrs|year|yr)|(\d+)\s*[-\s]?\s*(year|years|yrs|yr)'
                    name_match = re.match(NAME_REGEX, ' '.join(text_output.split()[:2]))
                    if name_match:
                       name = name_match.group()
                    else:
                        name = 'None'
                    phone_match = re.search(PHONE_REGEX, text_output)
                    if phone_match:
                        phone = phone_match.group()
                    else:
                        phone = 'None'
                    email_match = re.search(EMAIL_REGEX, text_output)
                    if email_match:
                        email = email_match.group()
                    else:
                        email = 'None'
                    experience_match = re.search(EXPERIENCE_REGEX, text_output)
                    if experience_match:
                        experience = experience_match.group()
                    else:
                        experience='Fresher'
                    new_df_data = {
                        'Name': [name],
                        'Phone No': [phone],
                        'Email': [email],
                        'Experience': [experience],
                        "JD Skills": [', '.join(self.jd_skills_normalized) if self.jd_skills_normalized else 'None'],
                        "Resume Skills": [', '.join(self.resume_skills_normalized) if self.resume_skills_normalized else 'None'],
                        "Matching Skills between JD & Resume": [', '.join(self.matching_skills) if self.matching_skills else 'None'],
                        "Cosine Similarity": [f"{self.similarity_score * 100:.2f}%" if 'similarity_score' in globals() else 0]
                    }
                    df_detail = pd.DataFrame(new_df_data)
                    file_path = r"D:\LLM_ALL_COLLAB_FOLDERS_freecodecamp_\prathmesh_GenAI_PROJECTS\Resume_Parsing NLP+Gen AI PROJECT\recruitment_data.csv"  #set csv path here
                    df_detail.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)
                    st.success(f"Data has been saved to Recruitment_CSV file successfully.")
            except Exception as e:
               st.error(f"Error saving data to CSV: {e}")

if __name__ == '__main__':
    ats_system = ATS_System()
    ats_system.run()