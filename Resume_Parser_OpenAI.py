from dotenv import load_dotenv
import base64
import streamlit as st
import os
import re
import io
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
import zipfile
import openai
from crewai import Agent, Task, Crew, Process



st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API key not loaded. Check your .env file for the GEMINI_API_KEY variable.")
else:
    genai.configure(api_key=api_key)

# Custom CSS for Styling
st.markdown(
    """
    <style>
        body {
            background-color: #1e272e;
            color: #dfe6e9;
            font-family: Arial, sans-serif;
        }
        .header-section {
            text-align: center;
            font-size: 48px;
            color: #FF4500;
            margin: 20px 0;
            padding: 10px;
            font-family: 'Roboto', sans-serif;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        }
        .stDataFrame {
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header-section">📄 ATS Tracking System 🔍</div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible&display=swap" rel="stylesheet">
    <h1 style="
        text-align: center;
        font-family: 'Atkinson Hyperlegible', sans-serif; 
        font-size: 35px; 
        color: #FFFFFF; 
        margin: 20px 0;
    ">
        ↪️ Analyze Your Resume for JD Compatibility and Similarity 
    </h1>
    <p style="
        text-align: center;
        font-family: 'Atkinson Hyperlegible', sans-serif; 
        font-size: 18px; 
        color: #FFFFFF; 
        line-height: 1.6;
        max-width: 700px;
        margin: auto;
    ">
Upload your Job Description and Resume, and get the Resume Analysis & Cosine Similarity score as per Job Description."</p>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------For Cover Letter Generator google-T5 model-------------------------------------------------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer


# Define the paths to the ZIP files
import zipfile
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the paths to the ZIP file and extraction directory
zip_file_path = r"D:\LLM_ALL_COLLAB_FOLDERS_freecodecamp_\prathmesh_GenAI_PROJECTS\Resume_Parsing NLP+Gen AI PROJECT\Prathmesh_google_t5-small_fine_tuned_cover_letter_model_v1.zip"
extracted_model_path = r"D:\LLM_ALL_COLLAB_FOLDERS_freecodecamp_\prathmesh_GenAI_PROJECTS\Resume_Parsing NLP+Gen AI PROJECT\extracted_model_directory" # Correct directory to extract the files

# Ensure the extracted model path exists, otherwise create it
if not os.path.exists(extracted_model_path):
    os.makedirs(extracted_model_path)

# Check and print the extracted directory path for validation
print(f"Extracting files to: {extracted_model_path}")

# Unzip the model file
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_model_path)
    print("Files extracted successfully!")
except Exception as e:
    print(f"Error while extracting files: {e}")
    st.error(f"Error while extracting files: {e}") # Display error in Streamlit

# Load the model and tokenizer from the extracted folder
try:
    print(f"Attempting to load model from: {extracted_model_path}")
    model = T5ForConditionalGeneration.from_pretrained(extracted_model_path)
    print("Model loaded successfully.")
    tokenizer = T5Tokenizer.from_pretrained(extracted_model_path)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error while loading the model or tokenizer: {e}")
    print(f"Exception details: {e}")
    st.error(f"Error while loading the model or tokenizer: {e}") # Display error in Streamlit
    model = None # Set model to None if loading fails
    tokenizer = None # Set tokenizer to None if loading fails





RESUME_FOLDER = "user_resumes"
os.makedirs(RESUME_FOLDER, exist_ok=True)

RESUME_FOLDER_1 = "ATS_SCORE_ABOVE_55"
os.makedirs(RESUME_FOLDER_1, exist_ok=True)

RESUME_FOLDER_2 = 'ATS_SCORE_BELOW_55'
os.makedirs(RESUME_FOLDER_2, exist_ok=True)


# =---------------------------------------------------Function to process Resume PDF file--------------------------------------------------------------------


# OCR Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# File Processing Functions
def input_pdf_setup(uploaded_file):
    try:
        temp_path = os.path.join(RESUME_FOLDER, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        images = convert_from_path(temp_path)
        os.remove(temp_path)
        text = "".join([pytesseract.image_to_string(img) for img in images])
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# =---------------------------------------------------Function to process Resume DOCX file--------------------------------------------------------------------

# pip install python-docx
import docx

def input_docx_setup(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
        return text
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return None
    



def input_image_setup(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# We have to remove @ symbol in punctuatation to detect Email from resume otherwise it will not detect,So...>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def preprocess_text(text):
    text = text.replace(" @", "@").replace("@ ", "@")
    new_punctuation = list(punctuation)
    new_punctuation.remove('@')
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and (word not in new_punctuation)]
    new = ' '.join(filtered_tokens)
    return new.replace(" @ ", "@").replace('\'','')


# Save Uploaded File
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs(RESUME_FOLDER, exist_ok=True)
        file_path = os.path.join(RESUME_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving the file: {e}")
        return None


# -----------Save candidate Resumes to folder based on Cosine SImilarity SCores SO it made easy to Recruiters to check Resumes only similarity score >55 ------------------------------------------------------------------------

def ATS_RESUME_ABOVE_55(uploaded_file):
    try:
        # Ensure the folder exists
        os.makedirs(RESUME_FOLDER_1, exist_ok=True)

        # Define the path where the file will be saved
        file_path = os.path.join(RESUME_FOLDER_1, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer()) 
        return file_path
    except Exception as e:
        st.error(f"Error saving the file: {e}")
        return None


def ATS_RESUME_BELOW_55(uploaded_file):
    try:
        # Ensure the folder exists
        os.makedirs(RESUME_FOLDER_2, exist_ok=True)

        # Define the path where the file will be saved
        file_path = os.path.join(RESUME_FOLDER_2, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer()) 
        return file_path
    except Exception as e:
        st.error(f"Error saving the file: {e}")
        return None


# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# To know Spacy to extract skills
IT_SKILLS = [
    "python", "java", "c++", "problem-solving", "algorithm design", "debugging", "git", "unit testing", "html", "css","javascript", "react", "node.js", "express.js", "restful apis", "mongodb", "sql", "agile methodology", "ruby", "android", "ios", "react native", "swift", "kotlin", "flutter", "app design", "mobile ui/ux", "unity", "unreal engine", "game physics", "3d modeling", "multiplayer design", "ai for games", "docker", "kubernetes", "aws", "ci/cd", "infrastructure as code", "ansible", "jenkins", "cloud services", "microcontrollers", "rtos", "circuit design", 
    "hardware-software integration", "sensors", "iot", "python", "r", "machine learning", "data analysis", "sql", "pandas", "numpy", "data visualization", "deep learning", "statistics", "excel", "tableau", "power bi", "data cleaning", "statistical analysis", "report generation", "tensorflow", "pytorch", "keras", "data preprocessing", "model evaluation", "neural networks", "regression", "natural language processing", "reinforcement learning", "research papers", "algorithms", "data warehousing", "etl", "data modeling", "big data technologies", "hadoop", "spark", "etl pipelines", "threat analysis", "risk assessment", "penetration testing", "network security", "firewalls", "siem", "encryption", "ethical hacking", "kali linux", "metasploit", "nmap", "vulnerability assessment", "exploit writing", "web application security", "network security", "cryptography", "risk management", "security governance", "compliance", "incident management", "forensics", "malware analysis", "security tools", "soc", "aws", "azure", "google cloud", "cloud infrastructure design", "cloud security", "automation", "networking", "terraform", "ci/cd", "cloud platforms", "virtualization", "servers", "storage solutions", "tcp/ip", "routing and switching", "vpns", "wireless networks", "network protocols", "storage systems", "nas", "san", "data backup", "disaster recovery", "storage", "project management", "product management", "scrum master", "it manager", "chief technology officer (cto)", 
    "agile", "stakeholder management", "risk management", "budgeting", "team leadership", "strategic planning", "product lifecycle","software testing", "qa engineering", "automation testing", "performance testing", "test management", "selenium", "junit", "cucumber", "bug tracking", "test plans", "load testing", "ui testing", "unit testing", "regression testing", "manual testing", "security testing",  "network administration", "help desk", "it support", "voip", "networking", "linux", "windows", "mac os", "remote desktop", "troubleshooting", "hardware setup", "dns", "dhcp", "vpn", "firewalls", "tcp/ip", "remote access", "cloud support", "ip addressing", "routing", "switching", "it consulting", "business analysis", "systems analysis", "stakeholder management", "project management", "enterprise architecture", "it strategy", "change management", "risk management", "consulting methodology", "requirements gathering", "client communication", "business process improvement", "systems integration"
]


def extract_skills_from_text(text, skill_set):
    nlp = spacy.blank("en")  # Initialize blank Spacy model
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_set] # IT_SKILLS
    matcher.add("SKILLS", patterns)
    doc = nlp(text)
    matches = matcher(doc)
    return {doc[start:end].text for _, start, end in matches}


# -------------------------------------------------------Cosine SImilarity Calculation uisng TFIDF-VECTORIZER-----------------------------------------------
def calculate_similarity(jd_skills, matching_skills):
    # Convert to lowercase and preprocess for TF-IDF
    jd_text = preprocess_text(', '.join(jd_skills))
    matching_text = preprocess_text(', '.join(matching_skills))
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, matching_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def normalize_skills(skills):
    return {skill.strip().lower() for skill in skills}

#------------------------------------------------------Suggestion to user Based on SImilarity Score-----------------------------------------------------------------------------------------------------

def give_suggestion(similarity_score):
    if similarity_score >= 0.80:
        color = "#4CAF50"  # Green
        font = "Times New Roman, serif"
    elif similarity_score >= 0.60:
        suggestion = "Good match! You have many of the required skills. Consider applying with confidence."
        color = "#4CAF50"  
        font = "Times New Roman, serif"
    elif similarity_score >= 0.50:
        suggestion = "Decent match, but there are areas for improvement. You may want to consider applying with additional learning in relevant areas.😊"
        color = "#4CAF50"  # Orange
        font = "Times New Roman, serif"
    else:
        suggestion = "Your skills don't closely match the job requirements. Consider gaining experience in the areas where you're lacking to strengthen your application."
        color = "#F44336"  # Red
        font = "Times New Roman, serif"
    return suggestion, color, font




# ----------------------------------------------We will give feedback to user based on COsine similarity SCore-----------------------------------------------------------------------------------------------------------------

def give_suggestion_submit4(similarity_score):
    if similarity_score * 100 >= 70:
        suggestion_ = (
            "🌟 Great match! Your skills align very well with the job requirements. "
            "High chances of getting shortlisted! Recruiters might call you soon. 😊"
        )
        color_ = "#4CAF50"  # Green
        font_ = "Times New Roman, serif"
    elif similarity_score * 100 >= 50:
        suggestion_ = (
            "✅ Good match! You have many of the required skills. "
            "There is a fair chance of being shortlisted. Consider applying confidently! 😊"
        )
        color_ = "#4CAF50"  # Green
        font_ = "Times New Roman, serif"
    elif similarity_score*100 <50:
        suggestion_ = (
            "🟠 Decent match, but there are areas for improvement. "
            "Low chances of getting shortlisted unless you improve on the relevant skills. "
            "Consider gaining additional experience before applying. 😊"
        )
        color_ ="#F44336"  # Orange
        font_ = "Times New Roman, serif"
    return suggestion_,color_,font_


# ------------------------------------------------------------------------------------for Submit4 button------------------------------------------------------------



import pandas as pd

def display_skills_and_similarity_submit4(jd_text, resume_text):
    # Extract skills
    jd_skills = extract_skills_from_text(jd_text, IT_SKILLS)
    resume_skills = extract_skills_from_text(resume_text, IT_SKILLS)

    global jd_skills_normalized
    jd_skills_normalized = normalize_skills(jd_skills)
    global resume_skills_normalized
    resume_skills_normalized = normalize_skills(resume_skills)

    global matching_skills
    matching_skills = jd_skills_normalized.intersection(resume_skills_normalized)
    
    global similarity_score_1
    similarity_score_1 = 0 

    if not matching_skills:
        st.markdown(f"No matching skills found between JD and Resume.", unsafe_allow_html=True)
    else:
        # Calculate similarity score only if there are matching skills
        similarity_score_1 = calculate_similarity(jd_skills_normalized, matching_skills)
        suggestion_, color_, font_ = give_suggestion_submit4(similarity_score_1)
        st.markdown(f"{suggestion_}", unsafe_allow_html=True)
    # Create a DataFrame to display the results
    data = {
        "JD Skills": [', '.join(jd_skills_normalized)],
        "Resume Skills": [', '.join(resume_skills_normalized)],
        "Matching Skills between JD & Resume": [', '.join(matching_skills) if matching_skills else 'None'],
        "Cosine Similarity": [f"{similarity_score_1 * 100:.2f}%" if similarity_score_1 else '0%']
    }
    df = pd.DataFrame(data)
    st.markdown("""
    """, unsafe_allow_html=True)

    st.markdown(f"<h3 style='color:{"#F8F8F8"};font-family:{"Calibri, serif"};'>Cosine Similarity between Job Description Skills and Matching Resume Skills as per JD:-  {similarity_score_1 * 100:.2f}%</h4>", unsafe_allow_html=True)
    st.markdown(df.to_html(classes='dataframe', index=False), unsafe_allow_html=True)




#-----------------------------------------------------------------# Display Extracted Skills and Similarity For Sidebar Task------------------------------------------------------------------------

import pandas as pd

def display_skills_and_similarity(jd_text, resume_text):
    # Extract skills
    jd_skills = extract_skills_from_text(jd_text, IT_SKILLS)
    resume_skills = extract_skills_from_text(resume_text, IT_SKILLS)

    # Normalize skills for consistent matching
    global jd_skills_normalized
    jd_skills_normalized = normalize_skills(jd_skills)
    global resume_skills_normalized
    resume_skills_normalized = normalize_skills(resume_skills)

    # Find matching skills
    global matching_skills
    matching_skills = jd_skills_normalized.intersection(resume_skills_normalized)

    if not matching_skills:
        st.markdown(f"No matching skills found between JD and Resume.", unsafe_allow_html=True)
        global similarity_score
        similarity_score = 0 # Assign a default value when no matching skills

    else:
        # global similarity_score
        similarity_score = calculate_similarity(jd_skills_normalized, matching_skills)

        suggestion, color, font = give_suggestion(similarity_score)
        st.markdown(f"<h2 style='color:{color}; font-family:{font};'>{suggestion}</h2>", unsafe_allow_html=True)


    # Create a DataFrame to display the results
    data = {
        "Job Description Skills": [', '.join(jd_skills_normalized)],
        "Resume Skills": [', '.join(resume_skills_normalized)],
        "Matching Skills between Job Description & Resume": [', '.join(matching_skills) if matching_skills else 'None'],
        "Cosine Similarity": [f"{similarity_score * 100:.2f}%" if 'similarity_score' in globals() else '0%'] # Use a default of 0 if 'similarity_score' is not defined
    }

    df = pd.DataFrame(data)
    st.markdown("""
        <style>
            .dataframe {
                font-family: 'Times New Roman', serif;
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }
            .dataframe th, .dataframe td {
                text-align: left;
                padding: 8px;
                border: 1px solid #ddd;
            }
            .dataframe th {
                background-color: #03A9F4;
                color: black;
            }
            .dataframe tr:nth-child(even) {
                background-color: #FFEB3B;
            }
            .dataframe tr:hover {
                background-color: #ddd;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Display the DataFrame as an HTML table
    st.markdown(f"<h4 style='color:{"#F8F8F8"};font-family:{"Calibri, serif"};'>Cosine Similarity between Job Description Skills and Matching Resume Skills as per JD:-  {similarity_score * 100:.2f}%</h4>", unsafe_allow_html=True)

    st.markdown(df.to_html(classes='dataframe', index=False), unsafe_allow_html=True)    # st.write(df)  # Display the DataFrame in the Streamlit app




import json

def extract_jd_info(job_description):

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
    response = get_openai_response(None,job_description, input_prompt_jd_extraction)
    if response:
        try:
            jd_data = json.loads(response)
            return jd_data
        except json.JSONDecodeError:
            st.error("Error: Gemini response is not in JSON format.")
            return None
    else:
        return None




# -----------------------------------------------------Input Prompts for Gemini AI--------------------------------------------------------------------------------------

input_prompt1 = """
You are a Technical HR Manager. Evaluate the skills from the resume against the job description:
- Match the technical skills from the job description with the resume and list only the matching skills in 4 lines.
- Job Description Experience in bullet points [mention years of experience required as per JD, e.g., 5 years]
- Candidate Resume Experience in bullet points [mention candidate's experience in years, e.g., 3 years] (Extracted from resume if available)
- List only missing technical skills in bullet points in 4 lines , be concise.
- Provide a suggestion Whether candidate should apply for job based on JD (e.g-You can apply for this job 😊) in 10 words.
"""


input_prompt3 = """
You are an ATS expert. Evaluate the resume’s keyword match with the job description.
- Mention the ATS Score:- percentage to user in big title (Eg-ATS Score 85%)
- Provide for overall keyword matching skills percentage (e.g-75%) in bullet points.
- List missing keywords as per job description only concisely in bullet points.
Keep the response concise, around 40 words.
"""


input_prompt4 = """
You are an ATS expert. Evaluate the resume’s keyword match with the job description.
- Mention the ATS Score as per mention JD & Resume matching skills & experience:- percentage to user in big title (Eg-ATS Score 85%)
- If ATS SCore is greater than 60% then provide feedback to user (eg.-Recruiter might call you soon,etc)  as per your knowledge in medium heading.
- If ATS SCore less than 60 then provide feedback to candidate improvement in resume(eg.-Get some experience,Learn AI Skills,etc) as per your knowledge. 
- Provide for overall keyword matching skills percentage (e.g-75%) in bullet points.
- Give siggestion to candidate based on ATS SCore between JD & Resume matching skills,experience in 20 words in bullet point.
- List missing keywords as per job description only concisely in bullet points.
Keep the response concise, around 40 words .
"""



# -------------------------------------------Function to get response from OpenAI-----------------------------------------------


def get_openai_response(input_text, doc_content, prompt):
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



# -------------------------CrewAI AGENTS To generate Cover Letter based oon JD & Resume ----------------------------------------------

def generate_cover_letter(job_description, resume_content):

    # Define the Agent :- Set up 'Worker' who will do this Job
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
# Tools:- are important in extending the capabilities of CrewAI agents...
    
    # Define the Task:- Describe what agents needs to do & what expected results should be
    cover_letter_task = Task(
    description=(
        "Generate a concise and impactful cover letter based on the provided resume and job description."
        "The cover letter should be professional, personalized, and highlight the candidate's skills strengths"
        "\nInputs:\n- Resume: {resume}\n- Job Description: {job_description}"
    ),
    expected_output="A structured, concise cover letter with an Introduction, Key Strengths, and Closing.",
    agent=cover_letter_agent
)
    
    # Create the Crew :- Put evrything together(Task,Agents) Crew manages workflows smoothly.
    cover_letter_crew = Crew(
        agents=[cover_letter_agent],
        tasks=[cover_letter_task],
        process=Process.sequential
    )
    
    # Inputs to the Crew
    inputs = {
        'resume': resume_content,
        'job_description': job_description
    }
    
    # Generate Cover Letter
    try:
        result = cover_letter_crew.kickoff(inputs=inputs)
        task_output = cover_letter_task.output

        return task_output.raw
    
    except Exception as e:
        st.error(f"Error during cover letter generation: {str(e)}")

# --------------------------------------------------------------------------------------------------------------------


# Output placeholders
output_placeholder = st.empty()
response_placeholder = st.empty()

# Initialize session state variables if they don't exist
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -We will Put Job Description in Backend provided by HR & Candidate will see this Job Description on interface & Candidate will upload his Resume & 
# We will let him know ,If he meets criteria as per Job Description skills.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------



with st.sidebar:
        st.markdown(
    """
    <div style="background-color:#64B5F6; padding:1px; border-radius:1px;">
        <h2 style="color:black; text-align:center;"> ↪️Upload Your Resume to Check Your Chances of Getting Shortlisted for the Below Job Description:</h2>
    </div>
    """,
    unsafe_allow_html=True
)
        
        # st.header('Upload Resume to check Your chances of getting shortlisted of this Job Description below:-')
        job_description = """
        As a Data Scientist, you will be assisting with designing and developing cutting-edge AI/ML solutions across the organization.
        - Basic working experience in data analysis and ML modeling.
        - Understanding of machine learning and deep learning concepts.
        - Familiarity with computer vision, natural language processing, and anomaly detection.
        - Awareness of GenAI and its applications in the industry.
        - Proficiency in Python.
        - Familiarity with frameworks and libraries like scikit-learn, TensorFlow, PyTorch, pandas, etc.
        - Basic knowledge of REST services using Flask or FastAPI.
        - Experience with cloud platforms (e.g., AWS, GCP, Azure) is a plus.
        Qualifications:
        - Bachelor's degree in engineering or related field.
        """

        st.write(job_description)
        st.session_state.job_description_preprocess = preprocess_text(job_description)

        # File Uploader
        uploaded_file_4 = st.file_uploader(
            "Upload your RESUME...",
            type=["pdf", "docx", "jpg", "jpeg", "png"],
            key="file_uploader_4"
        )



        if st.button:
            if "processed_data_4" not in st.session_state:
                st.session_state.processed_data_4 = {}

            # Button for Processing Resume
            if st.button("SEND", key="sending_button_4"):
                with st.spinner("Processing your resume..."):
                    if uploaded_file_4.name not in st.session_state.processed_data_4:
                        # Process based on file type
                        if uploaded_file_4.type == "application/pdf":
                            doc_content = input_pdf_setup(uploaded_file_4)
                        elif uploaded_file_4.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            doc_content = input_docx_setup(uploaded_file_4)
                        elif uploaded_file_4.type in ["image/jpeg", "image/png", "image/jpg"]:
                            doc_content = input_image_setup(uploaded_file_4)
                        else:
                            st.error("Unsupported file type. Please upload a PDF, DOCX, or image file.")
                            st.stop()

                        preprocess_doc_content_4 = preprocess_text(doc_content)

                        # Save processed data in session state
                        st.session_state.processed_data_4[uploaded_file_4.name] = {
                            "job_description_preprocess": st.session_state.job_description_preprocess,
                            "preprocess_doc_content_4": preprocess_doc_content_4,
                            "doc_content": doc_content,
                        }

                    data = st.session_state.processed_data_4[uploaded_file_4.name]

                    # Display Skills and Similarity
                    if "preprocess_doc_content_4" in data and data["preprocess_doc_content_4"]:
                        display_skills_and_similarity_submit4(data["job_description_preprocess"], data["preprocess_doc_content_4"])

                        # Call Gemini API
                        response = get_openai_response(data["job_description_preprocess"], data["doc_content"], input_prompt4)
                        if response:
                            st.write("### Gemini API Response:")
                            st.write(response)
                        else:
                            st.error("Failed to get response from Gemini API.")
                    else:
                        st.error("Error processing document content or no content available.")



# ------------------------------------------------------------User will Put Job Description & Resume file in PDF/DOCX/img---------------------------------------------------------------------------------


# Main Logic
with st.container():
    input = st.text_input("Job Description:", key="input", placeholder="Enter the job description here...")

    if input:
        st.session_state.preprocess_JD = preprocess_text(input)

    else:
        st.error('Please Mention Job Description! ')


    uploaded_file = st.file_uploader("Upload your resume:", type=["pdf", "docx", "jpg", "jpeg", "png"])
    if uploaded_file:
        pass
    else:
        st.error('Please Upload your Resume..')

    col1, col2, col3 = st.columns(3)

    with col1:
        submit1 = st.button("Tell Me About the Resume")

    with col2:
        submit3 = st.button("Percentage Match")

    with col3:  # New button for generating the cover letter
        submit5 = st.button("Generate Cover Letter")


    if submit1 or submit3:
        if uploaded_file:
            save_uploaded_file(uploaded_file)
            if uploaded_file.type == "application/pdf":
                doc_content = input_pdf_setup(uploaded_file)
                preprocess_doc_content = preprocess_text(doc_content)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc_content = input_docx_setup(uploaded_file)
                preprocess_doc_content = preprocess_text(doc_content)
            elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                doc_content = input_image_setup(uploaded_file)
                preprocess_doc_content = preprocess_text(doc_content)
            else:
                st.error("Unsupported file type. Please upload a PDF, DOCX, or image file.")
            
            if preprocess_doc_content:
                if input:
                    if submit1:
                        display_skills_and_similarity(st.session_state.preprocess_JD, preprocess_doc_content)
                        response = get_openai_response(st.session_state.preprocess_JD, doc_content, input_prompt1)
                        st.write(response)
                    if submit3:
                        response = get_openai_response(st.session_state.preprocess_JD, doc_content, input_prompt3)
                        st.write(response)
                else:
                    st.error('Please Mention Job Description')


    # -----------------------------------------------Generate Cocer letter Section__------------------------------------------------------


    if submit5: # When the "Generate Cover Letter" button is clicked
            if uploaded_file:
                save_uploaded_file(uploaded_file)

                if uploaded_file.type == "application/pdf":
                    doc_content = input_pdf_setup(uploaded_file)
                    preprocess_doc_content = preprocess_text(doc_content)

                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc_content = input_docx_setup(uploaded_file)
                    preprocess_doc_content = preprocess_text(doc_content)

                elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                    doc_content = input_image_setup(uploaded_file)
                    preprocess_doc_content = preprocess_text(doc_content)

                else:
                    st.error("Unsupported file type. Please upload a PDF, DOCX, or image file.")

                if preprocess_doc_content:
                    if input:
                        ai_agents_cover_letter = generate_cover_letter(input,doc_content)
                        st.subheader("Generated Cover Letter :")
                        st.write(ai_agents_cover_letter)
                    else:
                        st.error("Please mention Job Description..")
                else:
                    st.error('Please Mention Job Description')



# -------------------------------------------------------------REGEX PATTERN MATCHING----------------------------------------------------------------------------------


    try:
        if submit1:
            text_output = preprocess_doc_content

            NAME_REGEX = r"^[A-Za-z]+(?: [A-Za-z]+)+$|\b[a-zA-Z]+\s+[a-zA-Z]+\b"   # Matches first and last name
            PHONE_REGEX = r"(\+91[\s\-]?)?(\d{5})[\s\-]?(\d{2,8})|(\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}"
            EMAIL_REGEX = r"\b[a-z0-9_.]{2,30}[@][gmail.com]{1,15}\b|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            EXPERIENCE_REGEX = r'(\d+\.?\d*)\s+(years|yrs|year|yr)|(\d+(\.\d+)?)\s*(years|yrs|year|yr)|(\d+)\s*[-\s]?\s*(year|years|yrs|yr)'

            name_match = re.match(NAME_REGEX, ' '.join(text_output.split()[:2]))  # First two words as name
            if name_match:
                global name
                name = name_match.group()
            else:
                name = 'None'
                

            # Extract phone number
            phone_match = re.search(PHONE_REGEX, text_output)
            if phone_match:
                global phone
                phone = phone_match.group()
            else:
                phone = 'None'

            # Extract email
            email_match = re.search(EMAIL_REGEX, text_output)
            if email_match:
                global email
                email = email_match.group() 
            else:
                email = 'None'

            # Extract experience
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
                "JD Skills": [', '.join(jd_skills_normalized) if 'jd_skills_normalized' in globals() else 'None'],
                "Resume Skills": [', '.join(resume_skills_normalized) if 'resume_skills_normalized' in globals() else 'None'],
                "Matching Skills between JD & Resume":  [', '.join(matching_skills) if 'matching_skills' in globals() and matching_skills else 'None'],
                "Cosine Similarity": [f"{similarity_score * 100:.2f}%" if 'similarity_score' in globals() else 0]
            }

            df_detail = pd.DataFrame(new_df_data)

            # Save the data to CSV and confirm
            file_path = r"D:\LLM_ALL_COLLAB_FOLDERS_freecodecamp_\prathmesh_GenAI_PROJECTS\Resume_Parsing NLP+Gen AI PROJECT\recruitment_data.csv"
            df_detail.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)

            # If saved successfully
            st.success(f"Data has been saved to Recruitment_CSV file successfully.")

    except Exception as e:
        st.error(f"Error saving data to CSV: {e}")
            
        

