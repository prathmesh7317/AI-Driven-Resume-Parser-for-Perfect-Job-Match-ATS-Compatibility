# OpenAI-Driven-Resume-Parser-for-Perfect-Job-Match-ATS-Compatibility

## **Problem Statement**:

Job seekers often struggle to create resumes that align with job descriptions (JDs), which can significantly reduce their chances of getting shortlisted. Traditional methods of resume analysis are often time-consuming and ineffective in providing tailored feedback. This project addresses the gap by using AI-powered resume parsing combined with cosine similarity and ATS scoring to automatically assess a resume’s fit for a job role. It offers job seekers instant insights into how well their resume matches the job description, highlights missing skills, and provides recommendations for improvement, enhancing their chances of securing interviews. This tool helps streamline the application process for both job seekers and recruiters.


## Project Overview:

This project allows job seekers to upload their resumes and match them against job descriptions (JD). Using OCR for text extraction and NLP for skill detection, it calculates a cosine similarity score and ATS score. Based on the scores, it provides feedback and categorizes resumes for potential shortlisting. Resumes with scores above 60% are saved in a specific folder for further HR review. The system also suggests resume improvements and exports extracted data to a CSV file. It's designed to help candidates enhance their resumes for better job opportunities. The project also includes a CSV export feature for HR analysis and provides OpenAI-powered feedback on improving resumes. Candidates with a similarity score above 60% are categorized as potential shortlists for further action.

## New Feature: AI-Powered Cover Letter Generation

In addition to resume parsing, this project now integrates AI agents for cover letter generation. Based on the job description (JD) and the parsed resume data, the system uses OpenAI GPT to generate personalized cover letters tailored to the job role. The AI agent analyzes both the resume and JD, extracting relevant skills, experiences, and keywords to create a compelling cover letter that highlights the candidate's qualifications. The generated cover letter can be easily customized and saved for submission.

## Key Benefits of Cover Letter Generation:

Tailored Content: The AI generates personalized cover letters that align with the job requirements, ensuring relevance and increasing the chances of getting noticed.
Time-Saving: Automatically generates a professional and impactful cover letter, saving candidates time and effort.
Integration with Resume Parsing: The cover letter generation is integrated with the resume matching system, ensuring consistency and relevance in the application process.

## Key Features:

#### Resume Upload: Support for multiple formats (PDF, DOCX, image).
#### OCR-Based Text Extraction and Skill Detection: Uses Spacy NLP to detect skills, experience, and contact information from resumes.
#### Cosine Similarity and ATS Score Calculation: Automatically calculates the relevance of the resume for the job.
#### Suggestions for Resume Improvement: Powered by OpenAI to enhance resumes based on the job description.
#### Categorization of Resumes: Resumes are categorized based on similarity scores, with scores above 60% marked for further review.
#### CSV Export: Exports key resume data (skills, experience, contact info) to a CSV file for HR analysis.
#### Job Description Matching and Feedback: Provides recommendations on how to improve resumes to align with job descriptions.
#### OpenAI-Powered Cover Letter Generation: Personalized cover letters generated using AI Agents, tailored to the job description and the candidate's resume.
