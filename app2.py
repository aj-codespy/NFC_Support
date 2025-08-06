import os
import json
import base64
import requests
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from google.generativeai.types import HarmBlockThreshold, HarmCategory
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
import faiss
import numpy as np
import pickle
from supabase import create_client, Client
import uvicorn
import sys # Added this import
import inspect # Added for inspect.isawaitable

# Load environment variables from .env file
load_dotenv()

# --- Supabase Initialization ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
    print("INFO: Supabase client initialized successfully.")
else:
    print("WARNING: Supabase credentials not found. Database tools will be disabled.")


# --- Initialize LLM with error handling ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

chat_llm = None # For CrewAI agents (LangChain compatible)
direct_genai_model = None # For direct genai.GenerativeModel calls (e.g., image analysis, embeddings)

try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Configure genai for direct embedding and model usage
    genai.configure(api_key=GOOGLE_API_KEY)

    # LangChain-compatible LLM for CrewAI agents
    chat_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY, # Pass key directly
        safety_settings=[
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    )

    # Direct GenerativeModel for raw API calls (e.g., image analysis)
    direct_genai_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config={"temperature": 0.2},
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    print("INFO: Gemini LLM initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize Google Generative AI: {e}")
    print("Please ensure your GOOGLE_API_KEY is correct and the `google-generativeai` package is up-to-date.")
    print("Exiting application due to uninitialized LLM.")
    sys.exit(1) # Exit the application if LLM fails to initialize


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Patient Monitoring Agentic Workflow",
    description="A multi-agent system for early detection and cause analysis of patient deterioration in ICUs."
)

# --- WebSocket Connection Manager ---
class ConnectionManager:
    """Manages active WebSocket connections for broadcasting alerts."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts a new WebSocket connection and adds it to the list."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"New client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Removes a WebSocket connection from the list."""
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Broadcasts a message to all connected clients."""
        print(f"Broadcasting message to {len(self.active_connections)} clients: {message}")
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except RuntimeError as e: # Handle WebSocket connection closed unexpectedly
                print(f"WARNING: Could not send to WebSocket connection: {e}. Removing.")
                self.active_connections.remove(connection)


# Create a single instance of the manager for the application
manager = ConnectionManager() # Moved outside if __name__ == "__main__"

# --- WebSocket Endpoint ---
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive by waiting for messages
            await websocket.receive_text() # This will block until a message is received or connection closes
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"ERROR: WebSocket error: {e}")
        manager.disconnect(websocket)


# --- Pydantic Models for API ---
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


# --- RAG Setup (FAISS and Gemini Embeddings) ---
FAISS_INDEX_FILE = "faiss_index.bin"
TEXT_CHUNKS_FILE = "text_chunks_data.pkl"

faiss_index_global: Optional[faiss.Index] = None
text_chunks_global: List[str] = []
embedding_model_name = 'models/embedding-001' # Gemini embedding model

def get_embedding(text: str) -> np.ndarray:
    """Generates an embedding for the given text using the specified Gemini model."""
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not configured. Cannot generate embeddings.")
        # Return a zero vector with the correct dimension (768 for embedding-001)
        return np.zeros(768).astype('float32')
    # Sanitize input
    text = text.replace("\n", " ").strip()
    if not text:
        # The embedding model has 768 dimensions
        return np.zeros(768).astype('float32')
    try:
        # Use direct_genai_model for embedding content
        response = genai.embed_content(model=embedding_model_name, content=text)
        return np.array(response['embedding']).astype('float32')
    except Exception as e:
        print(f"ERROR: Embedding generation failed: {e}")
        # Return a zero vector on failure to avoid crashing the whole process
        return np.zeros(768).astype('float32')

try:
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(TEXT_CHUNKS_FILE):
        print("INFO: Loading pre-computed FAISS index and text chunks...")
        faiss_index_global = faiss.read_index(FAISS_INDEX_FILE)
        with open(TEXT_CHUNKS_FILE, "rb") as f:
            text_chunks_global = pickle.load(f)
        print("INFO: FAISS index and text chunks loaded successfully.")
    elif GOOGLE_API_KEY: # Only attempt to create if API key is present
        print("INFO: Pre-computed index not found. Creating a new one...")
        medical_text = """
        Hypertension: Also known as high blood pressure, it's a long-term medical condition. Symptoms include headaches and shortness of breath.
        Diabetes Mellitus: A group of metabolic diseases in which there are high blood sugar levels over a prolonged period. Common symptoms are frequent urination, increased thirst, and unexplained weight loss.
        Chronic Obstructive Pulmonary Disease (COPD): A progressive lung disease. Symptoms often include shortness of breath and a persistent cough.
        Pneumonia: An infection that inflames air sacs in one or both lungs. Symptoms include cough, fever, chills, and difficulty breathing.
        Aortic Stenosis: A narrowing of the aortic valve opening. Symptoms include chest pain, fainting, and heart palpitations.
        """
        text_chunks_global = [chunk.strip() for chunk in medical_text.splitlines() if chunk.strip()]

        if text_chunks_global:
            print("INFO: Generating embeddings for text chunks...")
            # Ensure embeddings are generated before creating FAISS index
            chunk_embeddings = np.array([get_embedding(chunk) for chunk in text_chunks_global])
            
            # Check if embeddings are valid (not all zeros)
            if chunk_embeddings.size == 0 or np.all(chunk_embeddings == 0):
                raise RuntimeError("Failed to generate valid embeddings for FAISS index creation.")

            dimension = chunk_embeddings.shape[1]
            faiss_index_global = faiss.IndexFlatL2(dimension)
            faiss_index_global.add(chunk_embeddings)

            faiss.write_index(faiss_index_global, FAISS_INDEX_FILE)
            with open(TEXT_CHUNKS_FILE, "wb") as f:
                pickle.dump(text_chunks_global, f)
            print("INFO: New FAISS index and text chunks created and saved.")
        else:
            print("WARNING: No text chunks to create FAISS index.")
            faiss_index_global = None
            text_chunks_global = []
    else:
        print("WARNING: GOOGLE_API_KEY not found. Cannot create FAISS index.")
        faiss_index_global = None
        text_chunks_global = []
except Exception as e:
    print(f"ERROR: Failed to load or create FAISS index: {e}")
    faiss_index_global = None
    text_chunks_global = []


# --- Custom Tools for CrewAI ---
@tool
async def messaging_service_tool(patient_id: str, message: str, color: str) -> str:
    """
    Broadcasts a patient-specific alert message with a color code to all 
    connected web clients via WebSocket and prints it for logging.
    The color should be a valid CSS color name (e.g., 'red', 'orange', 'green').
    """
    print("\n" + "="*50)
    print(f"--- URGENT ALERT for Patient {patient_id} ---")
    print(f"Message: {message}")
    print(f"Color Code: {color}")
    print("="*50 + "\n")
    
    # Broadcast a JSON object containing the patient_id, the alert message, and the color
    await manager.broadcast(json.dumps({
        "patient_id": patient_id,
        "alert": message,
        "color": color,
        "auth": "nurse" # Example auth field
    }))
    
    return f"Alert for patient {patient_id} successfully sent and broadcasted."

@tool
def monitor_api_connector(patient_id: str) -> str:
    """Fetch the latest patient vital signs from the Supabase database."""
    if not supabase:
        print("TOOL_ERROR: Supabase client not initialized. Cannot fetch vitals.")
        return json.dumps({"error": "Supabase client not initialized."})
        
    try:
        print(f"TOOL_EXECUTION: Querying Supabase for latest vitals for patient {patient_id}.")
        # Fetches the most recent record for the patient from the 'patient_vitals' table
        # Assuming 'created_at' is a timestamp column for ordering by recency
        response = supabase.table("patient_vitals").select("*").eq("pid", patient_id).order("created_at", desc=True).limit(1).execute()
        
        if response.data:
            print(f"TOOL_SUCCESS: Vitals fetched from Supabase for patient {patient_id}.")
            return json.dumps(response.data[0])
        else:
            print(f"TOOL_WARNING: No vitals found in Supabase for patient {patient_id}.")
            return json.dumps({"error": f"No vital signs found for patient {patient_id}."})
            
    except Exception as e:
        error_message = f"Failed to fetch vitals from Supabase: {e}"
        print(f"TOOL_ERROR: {error_message}")
        return json.dumps({"error": error_message})


@tool
def data_cleaning_tool(raw_data: str) -> str:
    """Cleans and formats raw patient data."""
    try:
        print(f"TOOL_EXECUTION: Cleaning raw data: {raw_data[:50]}...")
        # For mock, simply pass through. In a real scenario, this would involve parsing and validation.
        cleaned_data = raw_data
        print("TOOL_SUCCESS: Data cleaned.")
        return cleaned_data
    except Exception as e:
        print(f"TOOL_ERROR: Data Cleaning Tool failed: {e}")
        return json.dumps({"error": f"Failed to clean data: {e}"})

@tool
def mock_predictive_model_api() -> str:
    """
    Calls an external ML model API to get a mortality prediction.
    NOTE: This tool uses a static payload for the API call as per current implementation.
    Output: A JSON string with the prediction from the model.
    """
    print(f"TOOL_EXECUTION: Calling External Predictive Model API (Mock).")
    
    url = "https://web-production-4c1c0.up.railway.app/predict_mortality/"
    data = {
      "patient_data": [
          { "Time": "00:00", "Parameter": "RecordID", "Value": 999903 },
          { "Time": "00:00", "Parameter": "Age", "Value": 65 },
          { "Time": "00:00", "Parameter": "Gender", "Value": 0 },
          { "Time": "00:00", "Parameter": "Height", "Value": 170.0 },
          { "Time": "00:00", "Parameter": "Weight", "Value": 70.0 },
          { "Time": "00:00", "Parameter": "Urine", "Value": 50.0 },
          { "Time": "00:00", "Parameter": "HR", "Value": 85 },
          { "Time": "00:00", "Parameter": "Temp", "Value": 37.5 },
          { "Time": "00:00", "Parameter": "NIDiasABP", "Value": 70 },
          { "Time": "00:00", "Parameter": "SysABP", "Value": 120 },
          { "Time": "00:00", "Parameter": "DiasABP", "Value": 80 },
          { "Time": "00:00", "Parameter": "pH", "Value": 7.38 },
          { "Time": "00:00", "Parameter": "PaCO2", "Value": 42 },
          { "Time": "00:00", "Parameter": "PaO2", "Value": 85 },
          { "Time": "00:00", "Parameter": "Platelets", "Value": 250 },
          { "Time": "00:00", "Parameter": "MAP", "Value": 95 },
          { "Time": "00:00", "Parameter": "K", "Value": 4.0 },
          { "Time": "00:00", "Parameter": "Na", "Value": 140 },
          { "Time": "00:00", "Parameter": "FiO2", "Value": 0.21 },
          { "Time": "00:00", "Parameter": "GCS", "Value": 15 },
          { "Time": "00:00", "Parameter": "ICUType", "Value": 1 }
      ],
      "model_choice": "lgbm"
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        api_result = response.text
        print(f"TOOL_SUCCESS: Predictive API call successful. Response: {api_result}")
        return api_result
    except requests.exceptions.RequestException as e:
        error_message = f"Predictive API call failed: {e}"
        print(f"TOOL_ERROR: {error_message}")
        return json.dumps({"error": error_message})
    
@tool
def shap_explainer_tool(prediction_data: str) -> str:
    """
    Uses a mock explainer to identify key factors for the prediction.
    NOTE: This tool uses a static payload for the API call as per current implementation.
    """
    print(f"TOOL_EXECUTION: Calling SHAP Explainer API (Mock).")
    
    url = "https://web-production-4c1c0.up.railway.app/explain_mortality_shap_post/"
    data = {
      "patient_data": [
        { "Time": "00:00", "Parameter": "RecordID", "Value": 999903 },
          { "Time": "00:00", "Parameter": "Age", "Value": 65 },
          { "Time": "00:00", "Parameter": "Gender", "Value": 0 },
          { "Time": "00:00", "Parameter": "Height", "Value": 170.0 },
          { "Time": "00:00", "Parameter": "Weight", "Value": 70.0 },
          { "Time": "00:00", "Parameter": "Urine", "Value": 50.0 },
          { "Time": "00:00", "Parameter": "HR", "Value": 85 },
          { "Time": "00:00", "Parameter": "Temp", "Value": 37.5 },
          { "Time": "00:00", "Parameter": "NIDiasABP", "Value": 70 },
          { "Time": "00:00", "Parameter": "SysABP", "Value": 120 },
          { "Time": "00:00", "Parameter": "DiasABP", "Value": 80 },
          { "Time": "00:00", "Parameter": "pH", "Value": 7.38 },
          { "Time": "00:00", "Parameter": "PaCO2", "Value": 42 },
          { "Time": "00:00", "Parameter": "PaO2", "Value": 85 },
          { "Time": "00:00", "Parameter": "Platelets", "Value": 250 },
          { "Time": "00:00", "Parameter": "MAP", "Value": 95 },
          { "Time": "00:00", "Parameter": "K", "Value": 4.0 },
          { "Time": "00:00", "Parameter": "Na", "Value": 140 },
          { "Time": "00:00", "Parameter": "FiO2", "Value": 0.21 },
          { "Time": "00:00", "Parameter": "GCS", "Value": 15 },
          { "Time": "00:00", "Parameter": "ICUType", "Value": 1 }
      ],
      "model_choice": "lgbm"
      
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        api_result = response.text
        print(f"TOOL_SUCCESS: SHAP API call successful. Response: {api_result}")
        return api_result
    except requests.exceptions.RequestException as e:
        error_message = f"SHAP API call failed: {e}"
        print(f"TOOL_ERROR: {error_message}")
        return json.dumps({"error": error_message})


@tool
def database_writer_tool(patient_id: str, full_report: str) -> str:
    """Saves patient reports and events to a mock database."""
    try:
        print(f"TOOL_EXECUTION: Writing final report to mock database for patient {patient_id}.")
        # In a real app, this would write to a proper database.
        # For now, we just print the report to simulate saving.
        # print(f"Report for {patient_id}:\n{full_report}")
        print(f"TOOL_SUCCESS: Event record successfully saved for patient {patient_id} to local mock database.")
        return "Event record successfully saved to local mock database."
    except Exception as e:
        print(f"TOOL_ERROR: Database Writer Tool failed for patient {patient_id}: {e}")
        return f"Error writing to database: {e}"

@tool
def medical_rag_search(query: str) -> str:
    """Searches medical knowledge base for relevant information using RAG and reranks results."""
    print(f"TOOL_EXECUTION: Performing Medical RAG Search for query: '{query[:50]}...'")
    if faiss_index_global is None or not text_chunks_global or chat_llm is None:
        error_msg = "RAG functionality is not available due to missing FAISS index, text chunks, or LLM."
        print(f"TOOL_ERROR: {error_msg}")
        return error_msg
    try:
        query_embedding = get_embedding(query)
        # Ensure query_embedding is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = faiss_index_global.search(query_embedding, k=5)
        docs = [text_chunks_global[i] for i in indices[0]]
        print(f"TOOL_SUCCESS: Initial RAG search retrieved {len(docs)} documents.")

        rerank_prompt = f"""You are a medical assistant. Re-rank the following document chunks based on their relevance to the query. Return only the top 3 most relevant chunks in their original form, without any extra commentary.
        Query: {query}
        Documents: {json.dumps(docs)}
        """
        reranked_response = chat_llm.invoke(rerank_prompt)
        reranked_docs_str = reranked_response.content
        print("TOOL_SUCCESS: Reranking complete.")
        return reranked_docs_str
    except Exception as e:
        print(f"TOOL_ERROR: Medical RAG Search failed: {e}")
        return f"Error during RAG search: {e}"

@tool
def analyze_medical_image_with_ocr(base64_image: str, prompt: str) -> str:
    """Analyzes medical images and extracts text using OCR capabilities."""
    print(f"TOOL_EXECUTION: Analyzing medical image with OCR for prompt: '{prompt[:50]}...'")
    if not prompt:
        return "No prompt provided for analysis."
    if direct_genai_model is None:
        return "Image analysis failed. Direct genai model not initialized."
    if not base64_image:
        return "No image provided for analysis."
    try:
        # The direct_genai_model.generate_content expects a list of parts
        # where image data is provided as a dict with mime_type and data.
        image_part = {"mime_type": "image/jpeg", "data": base64.b64decode(base64_image)}
        combined_prompt_parts = [f"Analyze the image for visual content and also extract all text from it. Respond with a single summary containing both. User prompt: {prompt}", image_part]
        
        response = direct_genai_model.generate_content(combined_prompt_parts)
        print(f"TOOL_SUCCESS: Image analysis and OCR complete.")
        return response.text
    except Exception as e:
        print(f"TOOL_ERROR: Analyze Medical Image with OCR failed: {e}")
        return f"Error analyzing image: {e}"

# --- CrewAI Agent Definitions ---
# Moved agent definitions outside the if __name__ == "__main__" block
data_ingestor_agent = Agent(
    role="Data Steward",
    goal="Fetch and clean patient vital signs. Output must not contain markdown.",
    backstory="You are a meticulous data engineer responsible for a reliable, clean data pipeline. You fetch data from patient monitors, clean it, and structure it for downstream analysis.",
    tools=[monitor_api_connector, data_cleaning_tool],
    llm=chat_llm, verbose=True, allow_delegation=False
)

predictive_agent = Agent(
    role="Risk Assessment Analyst",
    goal="Use patient data to get a mock mortality prediction and provide a mock explanation.",
    backstory="You are a skilled AI expert. You simulate calling an external predictive model service to get a mortality risk prediction, probability, and an explanation. You must call the `mock_predictive_model_api` tool.",
    tools=[mock_predictive_model_api],
    llm=chat_llm, verbose=True, allow_delegation=False
)

explainability_agent = Agent(
    role="Clinical Interpreter",
    goal="Explain predictions in a human-understandable way. Output must not contain markdown.",
    backstory="You are an expert at translating complex AI predictions into actionable clinical insights. You use XAI models to provide a full picture.",
    tools=[shap_explainer_tool],
    llm=chat_llm, verbose=True, allow_delegation=False
)

environment_context_agent = Agent(
    role='Environmental Contextualizer',
    goal='Integrate non-vital-sign data to enrich the analysis. Output must not contain markdown.',
    backstory="You are the eyes and ears outside of the vital signs. You query EHRs for recent medication changes, lab results, and procedures. Your insights prevent false alarms and add crucial context.",
    llm=chat_llm, verbose=True
)

medical_knowledge_agent = Agent(
    role="Medical Knowledge Integrator",
    goal="Provide medical context, map symptoms to diseases, and suggest general solutions. Output must not contain markdown.",
    backstory="You are an intelligent medical AI with vast knowledge. You use a medical knowledge base (RAG) to provide context, identify potential conditions, and suggest high-level, non-prescriptive solutions.",
    tools=[medical_rag_search],
    llm=chat_llm, verbose=True, allow_delegation=False
)

image_analysis_agent = Agent(
    role="Medical Image Analysis Specialist with OCR",
    goal="Analyze an uploaded medical image for findings and extract text. Output must not contain markdown.",
    backstory="You are an expert in medical imaging with a state-of-the-art multi-modal vision model. You can analyze images and provide a comprehensive report on findings, including OCR.",
    tools=[analyze_medical_image_with_ocr],
    llm=chat_llm, verbose=True, allow_delegation=False
)

alerting_agent = Agent( # Moved alerting_agent definition here
    role="Notification Manager",
    goal="Broadcast timely, patient-specific alerts.",
    backstory="Central communication hub for delivering concise, prioritized alerts.",
    tools=[messaging_service_tool],
    llm=chat_llm, verbose=True
)

structuring_agent = Agent(
    role="Final Medical Report Generator",
    goal="Synthesize all findings into a single, well-structured, concise JSON report. Output must be a valid JSON object and nothing else.",
    backstory="You are the final stage. You receive information from colleagues and integrate it into a clear, professional JSON format. You ensure the final output is a clean, readable, and valid JSON object.",
    llm=chat_llm, verbose=True, allow_delegation=False
)

# --- Task Definitions ---
# Moved task definitions outside the if __name__ == "__main__" block
task_ingest_data = Task(
    description='For patient {patient_id}, use the Monitor API Connector to fetch the latest vital signs, then use the Data Cleaning Tool to prepare them.',
    expected_output='A clean JSON string of the patient\'s vital signs.',
    agent=data_ingestor_agent
)

task_predict_mortality = Task(
    description="""Given the patient's data (patient_text_history: {patient_text_history}, symptoms: {symptoms}),
    use the `mock_predictive_model_api` tool to get a mortality prediction, probability, and explanation.
    NOTE: The `mock_predictive_model_api` currently uses a static payload for its API call,
    so the input patient data will not directly influence its mock output.""",
    expected_output="A JSON string containing the mortality prediction, probability of death, and an explanation.",
    agent=predictive_agent
)

task_analyze_cause = Task(
    description='Using the prediction data from the previous step, analyze and explain the key factors contributing to the patient\'s risk.',
    expected_output='A human-readable sentence explaining the cause of the risk.',
    agent=explainability_agent,
    context=[task_predict_mortality]
)

task_get_context = Task(
    description='For patient {patient_id}, query the EHR system to get additional clinical context, such as recent lab results.',
    expected_output='A JSON string containing relevant clinical context.',
    agent=environment_context_agent
)

task_medical_knowledge = Task(
    description="""Given the patient's history '{patient_text_history}' and symptoms '{symptoms}',
    use the Medical RAG Search tool to find relevant medical context, potential diseases, and general solution suggestions.""",
    expected_output="A concise summary of medical context, potential diseases, and general solution suggestions.",
    agent=medical_knowledge_agent
)

task_image_analysis = Task(
    description="""Analyze the provided medical image for visual findings and extract all text using OCR.
    The base64 encoded image is available as {base64_image}.
    Use the patient's history '{patient_text_history}' and symptoms '{symptoms}' as the prompt for the analysis tool.
    If no image is provided ({base64_image} is empty), output 'No image provided for analysis.'""",
    expected_output="A detailed report on the visual findings and extracted text from the medical image, or a message indicating no image was provided.",
    agent=image_analysis_agent
)

# MODIFIED TASK: The agent now generates the alert message and color based on real-time data.
# This task is handled by the alerting agent, which is not part of the final report synthesis.
# It uses the manager.broadcast tool.
task_trigger_alert = Task(
    description="""
    Analyze the results from the vital signs fetch (context 1) and the mortality prediction (context 2).
    Based on this data, you must perform two actions:
    1.  **Determine a color code for the alert's severity:**
        - If the mortality prediction contains 'High' or 'Critical', the color MUST be 'red'.
        - If the mortality prediction contains 'Medium' or 'Moderate', the color MUST be 'orange'.
        - Otherwise, the color MUST be 'green'.

    2.  **Generate a concise and informative alert message** for patient {patient_id}. The message MUST include the mortality prediction and the specific vital signs that are concerning.
        Example Message: 'Patient {patient_id} has a High Mortality Risk with SpO2 at 91% and Heart Rate at 115 bpm.'

    Finally, you MUST broadcast this information by calling the `messaging_service_tool` with all three required arguments: the `patient_id`, the generated `message`, and the determined `color`.
    """,
    expected_output='A confirmation that the data-driven alert was successfully sent with a color code.',
    agent=alerting_agent,
    context=[task_ingest_data, task_predict_mortality] # Give the agent context from previous tasks
)

# --- Crew Creation & FastAPI Endpoint ---
@app.post("/generate-report/{patient_id}", response_model=MedicalReportResponse)
async def generate_medical_report(patient_id: str, data: PatientDataRequest):
    print(f"API_CALL: Received request for patient ID: {patient_id}")

    if chat_llm is None: # Check chat_llm for agent-based operations
        raise HTTPException(status_code=503, detail="LLM not initialized. Check API key.")
    if faiss_index_global is None:
        print("WARNING: RAG FAISS vector store not initialized. RAG functionality will be limited.")

    try:
        crew_inputs = {
            "patient_id": patient_id,
            "patient_text_history": data.patient_text_history,
            "symptoms": data.symptoms,
            "base64_image": data.medical_image_base64 or "",
        }

        # The structuring task needs context from all relevant previous tasks
        structuring_task_context = [
            task_ingest_data,
            task_predict_mortality,
            task_analyze_cause,
            task_get_context,
            task_medical_knowledge,
            task_image_analysis
        ]

        # Define the final structuring task within the endpoint
        structuring_task_final = Task(
            description="""Synthesize all findings from vital signs, mortality prediction, explanation, EHR context,
            medical knowledge, and image analysis into a single, well-structured, and concise JSON report.
            The report must contain these keys: 'patient_summary', 'image_analysis', 'related_conditions',
            'treatment_recommendations', 'mortality_prediction', 'probability_of_death', 'prediction_explanation', 'disclaimer'.
            - patient_summary: A synthesis of the patient's history, symptoms, and vital signs data.
            - image_analysis: A concise summary of the visual findings and OCR text from the image, or 'No image provided for analysis.' if applicable.
            - related_conditions: A list of potential medical conditions based on the combined analysis.
            - treatment_recommendations: High-level, non-prescriptive recommendations, formatted as a single string.
            - mortality_prediction: The predicted mortality risk (e.g., "Low Mortality Risk").
            - probability_of_death: The numerical probability of death from the prediction model.
            - prediction_explanation: An explanation of the prediction.
            - disclaimer: State this is for informational purposes and not a substitute for professional medical advice.
            The output must be a valid JSON object with no markdown formatting or any other text.
            If input seems non-medical, the 'disclaimer' should state 'Invalid input. Please provide medical-related data.'
            and other fields should be 'N/A' or empty lists.""",
            agent=structuring_agent,
            context=structuring_task_context,
            expected_output="A valid JSON object representing the final report with no markdown."
        )

        # Create the crew with all agents and ALL relevant tasks
        medical_crew = Crew(
            agents=[
                data_ingestor_agent,
                predictive_agent,
                explainability_agent,
                environment_context_agent,
                medical_knowledge_agent,
                image_analysis_agent,
                alerting_agent, # Added alerting agent to the crew
                structuring_agent
            ],
            tasks=[
                task_ingest_data,
                task_predict_mortality,
                task_analyze_cause,
                task_get_context,
                task_medical_knowledge,
                task_image_analysis,
                task_trigger_alert, # Added task_trigger_alert to the crew's tasks
                structuring_task_final # Final task to synthesize everything
            ],
            process=Process.sequential,
            verbose=2
        )

        print("CREW_EXECUTION: Kicking off medical report generation crew.")
        # Use kickoff for asynchronous execution
        raw_crew_result = medical_crew.kickoff(inputs=crew_inputs) # Get raw result
        
        # Check if the result is awaitable
        if inspect.isawaitable(raw_crew_result):
            crew_result = await raw_crew_result
        else:
            # If it's not awaitable, it means it returned a direct value (likely a string)
            # This indicates the Crew did not execute asynchronously, which is unexpected
            # if async tools are present.
            print("WARNING: medical_crew.kickoff() returned a non-awaitable object. Treating as synchronous result.")
            crew_result = raw_crew_result # It's already the final string result

        print("CREW_EXECUTION: Crew workflow completed.")

        # Robust JSON parsing
        final_report_parsed = {}
        try:
            # Find the start and end of the JSON object
            json_start = crew_result.find('{')
            json_end = crew_result.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = crew_result[json_start:json_end]
                final_report_parsed = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON object found in the output.", crew_result, 0)
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON decoding failed. Raw output: {crew_result}")
            raise HTTPException(
                status_code=500,
                detail=f"LLM output format error. Could not parse final JSON report. Details: {e}. Raw output: {crew_result}"
            )

        # --- Data type conversion before Pydantic validation ---
        # This makes the app resilient to the LLM returning a list instead of a string for recommendations.
        recommendations = final_report_parsed.get('treatment_recommendations')
        if isinstance(recommendations, list):
            print("INFO: Converting 'treatment_recommendations' from list to string.")
            final_report_parsed['treatment_recommendations'] = "\n".join(map(str, recommendations))
        
        # Ensure probability_of_death is a float
        prob_death = final_report_parsed.get('probability_of_death')
        try:
            final_report_parsed['probability_of_death'] = float(prob_death)
        except (ValueError, TypeError):
            final_report_parsed['probability_of_death'] = 0.0 # Default to 0.0 if conversion fails

        print("API_SUCCESS: Report generated and parsed successfully.")
        return MedicalReportResponse(
            status="success",
            message="Medical report generated.",
            **final_report_parsed
        )
    except Exception as e:
        print(f"API_ERROR: An unexpected error occurred during report generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )

# --- Main execution block for Uvicorn ---
if __name__ == "__main__":
    print("SERVER_START: Attempting to start Uvicorn server.")
    # To run: uvicorn app:app --reload --port 8003 (assuming your file is named app.py)
    uvicorn.run("app2:app", host="0.0.0.0", port=8009, reload=True)
