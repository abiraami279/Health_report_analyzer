import os
import re
import pandas as pd
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

class MIMICAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.load_mimic_data()
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.rag_chain = None
        self.subject_id = None
        self.hadm_id = None

    def load_mimic_data(self):
        self.admissions = pd.read_csv(f"{self.data_dir}/ADMISSIONS_sorted.csv")
        self.patients = pd.read_csv(f"{self.data_dir}/PATIENTS_sorted.csv")
        self.diagnoses = pd.read_csv(f"{self.data_dir}/DIAGNOSES_ICD_sorted.csv")
        self.diagnosis_names = pd.read_csv(f"{self.data_dir}/D_ICD_DIAGNOSES.csv")
        self.labs = pd.read_csv(f"{self.data_dir}/LABEVENTS_sorted.csv")
        self.lab_items = pd.read_csv(f"{self.data_dir}/D_LABITEMS.csv")
        self.notes = pd.read_csv(f"{self.data_dir}/NOTEEVENTS_sorted.csv")
        self.prescriptions = pd.read_csv(f"{self.data_dir}/PRESCRIPTIONS_sorted.csv")

    def extract_text(self, file_path):
        if file_path.endswith(".pdf"):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self.extract_text_from_image(file_path)
        else:
            raise ValueError("Unsupported file format.")

    def extract_text_from_pdf(self, file_path):
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages)

    def extract_text_from_image(self, file_path):
        return pytesseract.image_to_string(Image.open(file_path))

    def extract_ids(self, text):
        subject_id_match = re.search(r'\bSUBJECT_ID\s*[:\-]?\s*(\d+)', text)
        hadm_id_match = re.search(r'\bHADM_ID\s*[:\-]?\s*(\d+)', text)

        self.subject_id = int(subject_id_match.group(1)) if subject_id_match else None
        self.hadm_id = int(hadm_id_match.group(1)) if hadm_id_match else None

    def build_docs(self, extracted_text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)
        uploaded_docs = [Document(page_content=chunk) for chunk in chunks]

        mimic_docs = []
        def add_docs_from_df(df, label):
            for _, row in df.iterrows():
                content = f"{label}: " + " | ".join(f"{col}: {val}" for col, val in row.items())
                mimic_docs.append(Document(page_content=content))

        if self.subject_id and self.hadm_id:
            add_docs_from_df(self.patients[self.patients['SUBJECT_ID'] == self.subject_id], "Patient Info")
            add_docs_from_df(
                self.admissions[
                    (self.admissions['SUBJECT_ID'] == self.subject_id) &
                    (self.admissions['HADM_ID'] == self.hadm_id)
                ],
                "Admission Info"
            )
            add_docs_from_df(
                self.diagnoses[self.diagnoses['HADM_ID'] == self.hadm_id].merge(
                    self.diagnosis_names, on='ICD9_CODE', how='left'
                ),
                "Diagnosis"
            )
            add_docs_from_df(
                self.labs[self.labs['HADM_ID'] == self.hadm_id].merge(
                    self.lab_items, on='ITEMID', how='left'
                ),
                "Lab Result"
            )
            add_docs_from_df(self.notes[self.notes['HADM_ID'] == self.hadm_id], "Clinical Note")
            add_docs_from_df(self.prescriptions[self.prescriptions['HADM_ID'] == self.hadm_id], "Prescription")

        all_docs = uploaded_docs + mimic_docs
        return all_docs

    def setup_rag_chain(self, docs):
        vectorstore = DocArrayInMemorySearch.from_documents(docs, self.embedding_model)
        retriever = vectorstore.as_retriever()

        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=pipe)

        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

    def process_file(self, file_path):
        text = self.extract_text(file_path)
        self.extract_ids(text)
        docs = self.build_docs(text)
        self.setup_rag_chain(docs)
        return text  

    def ask_question(self, query):
        if not self.rag_chain:
            return "❗ The system is not initialized yet."
        result = self.rag_chain({"query": query})
        return result["result"]
