import os
import json
import base64
import requests
import sys
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from typing import List, Optional

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from supabase import create_client, Client
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import google.genergenerativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

# Load environment variables from a .env file for local development
load_dotenv()

# --- Environment Variable and Client Initialization ---

# Supabase Initialization: Connects to your patient database
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
    print("INFO: Supabase client initialized successfully.")
else:
    print("WARNING: Supabase credentials not found. Database tools will be disabled.")

# Google Generative AI Initialization: Sets up the language model for all agents
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = None
if not GOOGLE_API_KEY:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables. Exiting.")
    sys.exit(1)
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            safety_settings=[
                {"category": cat, "threshold": HarmBlockThreshold.BLOCK_NONE}
                for cat in [
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                ]
            ],
        )
        print("INFO: Unified Gemini LLM initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize Google Generative AI: {e}. Exiting.")
        sys.exit(1)


# --- FastAPI App and WebSocket Setup ---
app = FastAPI(
    title="Patient Monitoring Agentic Workflow",
    description="A multi-agent system for early detection and cause analysis of patient deterioration in ICUs.",
)

class ConnectionManager:
    """Manages active WebSocket connections to broadcast real-time alerts."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"New client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        print(f"Broadcasting message to {len(self.active_connections)} clients: {message}")
        # Iterate over a copy of the list to safely handle disconnections during broadcast
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Pydantic Models for API Data Validation ---
class PatientDataRequest(BaseModel):
    patient_text_history: str
    symptoms: str
    medical_image_base64: Optional[str] = None

class MedicalReportResponse(BaseModel):
    status: str
    message: str
    patient_summary: Optional[str] = None
    image_analysis: Optional[str] = None
    related_conditions: Optional[List[str]] = None
    treatment_recommendations: Optional[str] = None
    mortality_prediction: Optional[str] = None
    probability_of_death: Optional[float] = None
    prediction_explanation: Optional[str] = None
    disclaimer: Optional[str] = None
    error_details: Optional[str] = None

# --- RAG Setup for Medical Knowledge Base ---
FAISS_INDEX_FILE = "faiss_index.bin"
TEXT_CHUNKS_FILE = "text_chunks_data.pkl"
faiss_index_global: Optional[faiss.Index] = None
text_chunks_global: List[str] = []
embedding_model_name = 'models/embedding-001'

def get_embedding(text: str) -> np.ndarray:
    text = text.replace("\n", " ").strip()
    if not text: return np.zeros(768).astype('float32')
    try:
        response = genai.embed_content(model=embedding_model_name, content=text)
        return np.array(response['embedding']).astype('float32')
    except Exception as e:
        print(f"ERROR: Embedding generation failed: {e}")
        return np.zeros(768).astype('float32')

try:
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(TEXT_CHUNKS_FILE):
        print("INFO: Loading pre-computed FAISS index and text chunks...")
        faiss_index_global = faiss.read_index(FAISS_INDEX_FILE)
        with open(TEXT_CHUNKS_FILE, "rb") as f: text_chunks_global = pickle.load(f)
    else:
        print("INFO: Pre-computed index not found. Creating a new one...")
        medical_text = """
        Hypertension: High blood pressure. Symptoms: headaches, shortness of breath.
        Diabetes Mellitus: High blood sugar. Symptoms: frequent urination, increased thirst.
        COPD: Progressive lung disease. Symptoms: shortness of breath, persistent cough.
        Pneumonia: Lung infection. Symptoms: cough, fever, chills, difficulty breathing.
        Aortic Stenosis: Narrowing of the aortic valve. Symptoms: chest pain, fainting.
        """
        text_chunks_global = [chunk.strip() for chunk in medical_text.splitlines() if chunk.strip()]
        if text_chunks_global:
            chunk_embeddings = np.array([get_embedding(chunk) for chunk in text_chunks_global])
            dimension = chunk_embeddings.shape[1]
            faiss_index_global = faiss.IndexFlatL2(dimension)
            faiss_index_global.add(chunk_embeddings)
            faiss.write_index(faiss_index_global, FAISS_INDEX_FILE)
            with open(TEXT_CHUNKS_FILE, "wb") as f: pickle.dump(text_chunks_global, f)
            print("INFO: New FAISS index and text chunks created and saved.")
except Exception as e:
    print(f"ERROR: Failed to load or create FAISS index: {e}")

# --- Custom Tools for CrewAI Agents ---
@tool
async def messaging_service_tool(patient_id: str, message: str, color: str) -> str:
    """Broadcasts a patient-specific alert with a color code to all web clients."""
    print(f"\n--- URGENT ALERT for Patient {patient_id} ---\nMessage: {message}\nColor Code: {color}\n" + "="*50)
    await manager.broadcast(json.dumps({"patient_id": patient_id, "alert": message, "color": color}))
    return f"Alert for patient {patient_id} successfully broadcasted."

@tool
def monitor_api_connector(patient_id: str) -> str:
    """Fetch the latest patient vital signs from the Supabase database."""
    if not supabase: return json.dumps({"error": "Supabase client not initialized."})
    try:
        response = supabase.table("patient_vitals").select("*").eq("pid", patient_id).order("id", desc=True).limit(1).execute()
        if response.data: return json.dumps(response.data[0])
        return json.dumps({"error": f"No vital signs found for patient {patient_id}."})
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch vitals from Supabase: {e}"})

@tool
def predictive_model_api(patient_data_json: str) -> str:
    """Calls an external ML model API to get a mortality prediction."""
    url = "https://web-production-4c1c0.up.railway.app/predict_mortality/"
    # The tool dynamically creates the payload from the input vitals data
    vitals = json.loads(patient_data_json)
    patient_data_list = [{"Parameter": key, "Value": value} for key, value in vitals.items()]
    data = {"patient_data": patient_data_list, "model_choice": "lgbm"}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Predictive API call failed: {e}"})

@tool
def shap_explainer_tool(patient_data_json: str) -> str:
    """Calls an external API to get a SHAP explanation for a prediction."""
    url = "https://web-production-4c1c0.up.railway.app/explain_mortality_shap_post/"
    vitals = json.loads(patient_data_json)
    patient_data_list = [{"Parameter": key, "Value": value} for key, value in vitals.items()]
    data = {"patient_data": patient_data_list, "model_choice": "lgbm"}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"SHAP API call failed: {e}"})

@tool
def medical_rag_search(query: str) -> str:
    """Searches medical knowledge base using RAG."""
    if faiss_index_global is None: return "RAG system not available."
    try:
        query_embedding = get_embedding(query)
        _, indices = faiss_index_global.search(np.array([query_embedding]), k=3)
        docs = [text_chunks_global[i] for i in indices[0]]
        return "\n".join(docs)
    except Exception as e:
        return f"Error during RAG search: {e}"

@tool
def analyze_medical_image_with_ocr(base64_image: str, prompt: str) -> str:
    """Analyzes medical images using OCR."""
    if not base64_image: return "No image provided for analysis."
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Analyze the image for visual content and text. User prompt: {prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        response = llm.invoke(message)
        return response.content
    except Exception as e:
        return f"Error analyzing image: {e}"

# --- CrewAI Agent & Task Definitions ---
data_ingestor_agent = Agent(role="Data Steward", goal="Fetch patient vital signs.", backstory="Meticulous data engineer.", tools=[monitor_api_connector], llm=llm, verbose=True)
predictive_agent = Agent(role="Risk Assessment Analyst", goal="Get a mortality prediction from the external API.", backstory="AI expert calling a predictive model service.", tools=[predictive_model_api], llm=llm, verbose=True)
explainability_agent = Agent(role="Clinical Interpreter", goal="Get a SHAP explanation for the prediction.", backstory="Expert translating AI predictions.", tools=[shap_explainer_tool], llm=llm, verbose=True)
medical_knowledge_agent = Agent(role="Medical Knowledge Integrator", goal="Provide medical context for symptoms.", backstory="AI with vast medical knowledge using a RAG system.", tools=[medical_rag_search], llm=llm, verbose=True)
image_analysis_agent = Agent(role="Medical Image Analyst", goal="Analyze an uploaded medical image.", backstory="Expert in medical imaging with OCR capabilities.", tools=[analyze_medical_image_with_ocr], llm=llm, verbose=True)
alerting_agent = Agent(role="Notification Manager", goal="Broadcast timely, patient-specific alerts.", backstory="Central communication hub for delivering alerts.", tools=[messaging_service_tool], llm=llm, verbose=True)
structuring_agent = Agent(role="Final Report Generator", goal="Synthesize all findings into a single, raw JSON object.", backstory="Final stage, integrating all info into a clean JSON format.", llm=llm, verbose=True)

# --- FastAPI Endpoint ---
@app.post("/generate-report/{patient_id}", response_model=MedicalReportResponse)
async def generate_medical_report(patient_id: str, data: PatientDataRequest):
    if not llm: raise HTTPException(status_code=503, detail="LLM not initialized.")
    
    crew_inputs = {
        "patient_id": patient_id,
        "patient_text_history": data.patient_text_history,
        "symptoms": data.symptoms,
        "base64_image": data.medical_image_base64 or "",
    }

    # Define all tasks for the workflow
    task_ingest_data = Task(description=f"For patient {patient_id}, fetch the latest vital signs.", expected_output='A clean JSON string of vital signs.', agent=data_ingestor_agent)
    task_predict_mortality = Task(description=f"Using the fetched vitals, call the predictive model API.", expected_output="JSON string with prediction details.", agent=predictive_agent, context=[task_ingest_data])
    task_explain_prediction = Task(description=f"Based on the fetched vitals, call the SHAP explainer tool.", expected_output="A string containing the SHAP explanation.", agent=explainability_agent, context=[task_ingest_data])
    task_rag_search = Task(description=f"Based on the patient's symptoms: '{data.symptoms}', search for related medical conditions.", expected_output="A list of potential related conditions.", agent=medical_knowledge_agent)
    task_analyze_image = Task(description=f"Analyze the provided image using the patient's history as a prompt: '{data.patient_text_history}'.", expected_output="A textual description of the image analysis.", agent=image_analysis_agent)
    task_trigger_alert = Task(
        description=f"""Analyze the vital signs and mortality prediction for patient {patient_id}.
        1. Determine a color: 'red' for high risk, 'orange' for medium, 'green' for low.
        2. Generate a concise alert message including the risk and key vitals.
        3. Call the messaging tool with the patient_id, generated message, and color.""",
        expected_output='A confirmation that the alert was sent.',
        agent=alerting_agent,
        context=[task_ingest_data, task_predict_mortality]
    )
    structuring_task_final = Task(
        description="Synthesize all findings (vitals, prediction, explanation, RAG search, image analysis) into a single JSON report.",
        expected_output="A valid, raw JSON object representing the final report.",
        agent=structuring_agent,
        context=[task_ingest_data, task_predict_mortality, task_explain_prediction, task_rag_search, task_analyze_image]
    )

    medical_crew = Crew(
        agents=list(set([task.agent for task in [task_ingest_data, task_predict_mortality, task_explain_prediction, task_rag_search, task_analyze_image, task_trigger_alert, structuring_task_final]])),
        tasks=[task_ingest_data, task_predict_mortality, task_explain_prediction, task_rag_search, task_analyze_image, task_trigger_alert, structuring_task_final],
        process=Process.sequential,
        verbose=2
    )
    
    try:
        print("CREW_EXECUTION: Kicking off medical report generation crew.")
        crew_result = await medical_crew.kickoff_async(inputs=crew_inputs)
        print("CREW_EXECUTION: Crew workflow completed.")

        json_start = crew_result.find('{')
        json_end = crew_result.rfind('}') + 1
        if json_start == -1: raise ValueError("No JSON object found in the final output from the crew.")
        json_str = crew_result[json_start:json_end]
        final_report_parsed = json.loads(json_str)

        if isinstance(final_report_parsed.get('treatment_recommendations'), list):
            final_report_parsed['treatment_recommendations'] = "\n".join(map(str, final_report_parsed['treatment_recommendations']))
        
        try:
            final_report_parsed['probability_of_death'] = float(final_report_parsed.get('probability_of_death'))
        except (ValueError, TypeError):
            final_report_parsed['probability_of_death'] = 0.0

        return MedicalReportResponse(status="success", message="Medical report generated.", **final_report_parsed)

    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"LLM output format error: {e}. Raw output: {crew_result}")
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Final report validation failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Main execution block for running the server ---
if __name__ == "__main__":
    # Use the PORT environment variable provided by Railway, default to 8003 for local dev
    port = int(os.environ.get("PORT", 8003))
    # Use "main:app" if you save the file as main.py
    uvicorn.run("main:app", host="0.0.0.0", port=port)
