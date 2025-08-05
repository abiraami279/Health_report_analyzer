from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import re

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "".join(page.extract_text() or "" for page in reader.pages)
    else:
        image = Image.open(file)
        return pytesseract.image_to_string(image)

def extract_ids_from_text(text):
    subject_id = re.search(r'\\bSUBJECT_ID\\s*[:\\-]?\\s*(\\d+)', text)
    hadm_id = re.search(r'\\bHADM_ID\\s*[:\\-]?\\s*(\\d+)', text)
    return (
        int(subject_id.group(1)) if subject_id else None,
        int(hadm_id.group(1)) if hadm_id else None
    )

def run_rag_pipeline(query, text, subject_id, hadm_id):
    from model.mimic_model import build_qa_chain
    rag_chain = build_qa_chain(text, subject_id, hadm_id)
    result = rag_chain({"query": query})
    return result["result"]
