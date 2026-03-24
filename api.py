import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import webbrowser
import threading
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import csv
import os

from main import MediAIHospitalSystem

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open browser slightly after server startup
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    yield

# Create the FastAPI app
app = FastAPI(title="MediAI Hospital System UI", lifespan=lifespan)

# Initialize the global system instance
system = MediAIHospitalSystem()

# Mount the static directory to serve index.html, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define our expected incoming data model
class PatientData(BaseModel):
    id: str
    age: int
    gender: str
    heart_rate: int
    blood_pressure_systolic: int
    temperature: float
    symptoms_text: str

@app.get("/")
async def root():
    return FileResponse('static/index.html')

def record_to_csv(patient_id, age, gender, symptoms, diagnosis_list, explanation, treatment, risk):
    filename = 'patient_records.csv'
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Patient ID', 'Age', 'Gender', 'Symptoms', 'Diagnosis', 'Explanation', 'Treatment', 'Mortality Risk']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'Patient ID': patient_id,
            'Age': age,
            'Gender': gender,
            'Symptoms': symptoms,
            'Diagnosis': ', '.join(diagnosis_list) if isinstance(diagnosis_list, list) else diagnosis_list,
            'Explanation': explanation,
            'Treatment': treatment,
            'Mortality Risk': f"{risk:.4f}"
        })

@app.post("/api/emergency")
async def process_emergency(patient: PatientData):
    try:
        # convert back to dict for the main sys
        patient_dict = {
            'id': patient.id,
            'age': patient.age,
            'gender': patient.gender,
            'heart_rate': patient.heart_rate,
            'blood_pressure_systolic': patient.blood_pressure_systolic,
            'temperature': patient.temperature,
            'symptoms_text': patient.symptoms_text,
            'xray_image': None  # Skipping X-Ray for the web form for now
        }
        
        logger.info(f"Received emergency API call for patient {patient.id}")
        
        # Await the system's emergency handler
        result = await system.handle_emergency(patient_dict)
        
        # Format treatments nicely for the frontend
        assignments = []
        if result.get('treatment_plan') and hasattr(result['treatment_plan'], 'assignments'):
            for assignment in result['treatment_plan'].assignments:
                assignments.append({
                    'resource': assignment[0],
                    'task': assignment[1],
                    'time_hours': float(assignment[2])
                })
        
        response_data = {
            'patient_id': result['patient_id'],
            'diagnosis': result['diagnosis'].get('diagnoses', []),
            'explanation': result['diagnosis'].get('explanation', 'No explanation available'),
            'mortality_risk': result['mortality_risk'],
            'treatment_assignments': assignments,
            'full_report': result['report']
        }
        
        # Log to spreadsheet
        treatments_str = ", ".join([f"{a['task']} by {a['resource']}" for a in assignments]) if assignments else str(result.get('treatment_plan', 'None'))
        record_to_csv(
            patient_id=result['patient_id'],
            age=patient.age,
            gender=patient.gender,
            symptoms=patient.symptoms_text,
            diagnosis_list=result['diagnosis'].get('diagnoses', []),
            explanation=result['diagnosis'].get('explanation', 'None'),
            treatment=treatments_str,
            risk=result['mortality_risk']
        )
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing emergency: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
