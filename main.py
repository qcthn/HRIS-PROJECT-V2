import os
import json
import pdfplumber
import pandas as pd
import shutil
from docx import Document
import re
from fastapi import FastAPI, HTTPException, Query,Header
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load API Key
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("Thiếu OPENAI_API_KEY trong .env")
# client = OpenAI(api_key=api_key)

# Setup FastAPI
app = FastAPI()

# Folders
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning function
def clean_text(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text).strip()

# File extraction function
def extract_text_from_file(file_path: str) -> str:
    ext = file_path.split('.')[-1].lower()
    
    try:
        if ext == "pdf":
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(clean_text(page.extract_text() or "") for page in pdf.pages)
        elif ext == "docx":
            doc = Document(file_path)
            return "\n".join(clean_text(p.text) for p in doc.paragraphs)
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return clean_text(f.read())
        elif ext == "xlsx":
            df = pd.read_excel(file_path, dtype=str)
            return clean_text(df.to_string(index=False))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý {ext.upper()}: {e}")
    
    raise HTTPException(status_code=400, detail="Định dạng không hỗ trợ")

# ✅ Updated: Correctly handle multiple file paths in query parameters
@app.get("/get-extracted-text/")
async def get_extracted_text(file_paths: List[str] = Query(...)):
    """Extracts text from multiple files and returns combined text."""
    extracted_texts = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy {file_path}")
        extracted_texts.append(extract_text_from_file(file_path))

    return {"extracted_text": "\n\n".join(extracted_texts)}
# ✅ Xử lý API Key do người dùng cung cấp
def get_openai_client(user_api_key: str):
    """Tạo OpenAI client sử dụng API Key do người dùng cung cấp."""
    if not user_api_key:
        raise HTTPException(status_code=400, detail="Bạn cần cung cấp OpenAI API Key.")
    return OpenAI(api_key=user_api_key)
# OpenAI JSON extraction
def clean_json_response(response_text):
    """Làm sạch phản hồi GPT để loại bỏ các ký tự không mong muốn trước khi phân tích JSON."""
    response_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', response_text)
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()
def extract_info_with_openai(text: str,user_api_key: str) -> dict:
    prompt = f"""
    Trích xuất thông tin từ văn bản CV và trả về JSON hợp lệ:
    {{
        "Name": "",
        "Email": "",
        "Phone": "",
        "Skills": [],  
        "Experience": [],  
        "Education": [],  
        "Certifications": [],  
        "Languages": [],  
        "Strengths": [],  
        "Weaknesses": [],  
        "Additional information": []
    }}
    For the **Languages** field, include:
    - The candidate's native language based on their nationality (e.g., Vietnamese for a candidate from Vietnam).
    - Any foreign language certifications (e.g., TOEIC score) and the corresponding language proficiency level (e.g., English with a proficiency level based on the score).

    For **Strengths and Weaknesses**, analyze the candidate's work experience to identify:
    - **Strengths:** Key skills and attributes demonstrated through their experience.
    - **Weaknesses:** Areas for improvement or challenges faced in their roles.
    Văn bản CV:
    {text}
    """
    client = get_openai_client(user_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
             {"role": "system", "content": "You are an expert in extracting information from CVs (resumes) and images with 10 years of experience in getting the exact information needed to recruit suitable positions for the company."

            "**Context:** I will provide you with resumes of candidates (which can be one or more) or image files containing text."

            "**Your task** is to extract information from the resumes and images I provide (I have taken the text from the resume, and the image will be provided to you below) and return the output as a JSON file."

            "Some of the most important information required for each candidate includes:"
            "- Name"
            "- Email"
            "- Phone number"
            "- Skills"
            "- Experience (including: position, timeline, responsibilities)"
            "- Education (including: degree, institution, timeline, GPA)"
            "- Certifications"
            "- Languages (including proficiency based on nationality and language certifications)"
            "- Strengths (based on the candidate's experience and job description)"
            "- Weaknesses (based on the candidate's experience and job description)"
            "- Additional information (including identification and visa details if provided)"

            "**Task:** Extract the following information from the CV text and return it as JSON."

            "**Output:** JSON file format"

            "***Note:** I can provide you with the text, but in that text will be a synthesis of many resumes of different candidates.*"

            "**REMEMBER:** The output should only be in JSON format."},
            {"role": "user", "content": prompt}]
    )
    extracted_text = response.choices[0].message.content.strip()
    cleaned_text = clean_json_response(extracted_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích JSON: {e}")

# ✅ Updated: Correctly handle multiple file paths in query parameters
@app.get("/get-json/")
async def get_json(
    file_paths: List[str] = Query(...),
    openai_api_key: str = Header(None, convert_underscores=False)):
    """Extracts text from files, processes into JSON, and returns JSON file."""
    if not openai_api_key:
        raise HTTPException(status_code=400, detail="Thiếu OpenAI API Key. Vui lòng cung cấp qua 'openai_api_key' trong header.")
    extracted_texts = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy {file_path}")
        extracted_texts.append(extract_text_from_file(file_path))

    full_text = "\n\n".join(extracted_texts)
    extracted_info = extract_info_with_openai(full_text,openai_api_key)

    json_file = os.path.join(OUTPUT_DIR, "combined_output.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(extracted_info, f, ensure_ascii=False, indent=4)

    return FileResponse(json_file, media_type="application/json", filename="combined_output.json")

# ✅ Updated: Ensure correct naming for file parameter
class QuestionRequest(BaseModel):
    filename: str
    question: str

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest,
                    openai_api_key: str = Header(None, convert_underscores=False)):
    """Extracts text from a file and answers a question using OpenAI."""
    if not openai_api_key:
        raise HTTPException(status_code=400, detail="Thiếu OpenAI API Key. Vui lòng cung cấp qua 'openai_api_key' trong header.")
    if not os.path.exists(request.filename):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy {request.filename}")

    extracted_text = extract_text_from_file(request.filename)
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Play the role of a professional HR, with 10 years of experience in finding potential candidates suitable for the company based on the CV (resume) they send Context: I will provide you with information of each CV (resume) in text form, from which I will ask you some questions related to the CV (resume) of this candidate Task: Please provide the most accurate and closest information to the question I asked, helping me have the most objective view of this candidate so that I can decide whether to hire him or not Tone: solemn, dignified, straightforward, suitable for the office environment, recruitment. Below is the content of the candidate's CV\n{extracted_text}"},
                {"role": "user", "content": request.question}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
        # return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi OpenAI API: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))  # Lấy PORT từ Heroku hoặc mặc định 8000 nếu chạy local
    uvicorn.run(app, host="0.0.0.0", port=port)
