from supabase import create_client, Client
import os
from dotenv import load_dotenv
import requests
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
id='9c8be425-e2fc-42bd-bcc7-9db5d966af42'
response = supabase.table("patient_vitals").select("*").eq("pid", id).limit(1).execute()
print(response)

def mock_predictive_model_api() -> str:
    """
    Calls an external ML model API to get a mortality prediction.
    Input: A JSON string of patient features (currently ignored, uses a template).
    Output: A JSON string with the prediction from the model.
    """
    print(f"TOOL_EXECUTION: Calling External Predictive Model API.")
    
    # The URL for the external prediction service
    url = "https://web-production-4c1c0.up.railway.app/predict_mortality/"

    # NOTE: This is a static payload based on your example. A more advanced
    # implementation would require another agent or a complex tool to parse
    # the unstructured 'patient_text_history' and 'symptoms' into this
    # structured format. For now, we use this template to ensure the API call works.
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
    response = requests.post(url, json=data)
    print(response.text)
    


def shap_explainer_tool() -> str:
    """Uses a mock explainer to identify key factors for the prediction."""
    
    url = "https://web-production-4c1c0.up.railway.app//explain_mortality_shap/"
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
    response = requests.post(url, json=data)
    #print(response.text)
    
mock_predictive_model_api()
shap_explainer_tool()
