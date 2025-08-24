import os
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from google.cloud import storage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from huggingface_hub import login
from google.cloud import secretmanager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# PART 1: AUTHENTICATION
# ==============================================================================

def access_secret_version(project_id, secret_id, version_id="latest"):
    """Accesses a secret version from Google Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Failed to access secret {secret_id}: {e}")
        raise

# Configuration
PROJECT_ID = "premium-cipher-462011-p3"
SECRET_ID = "HF_TOKEN"
BUCKET_NAME = "mlops-course-premium-cipher-462011-p3-unique"
GCS_MODEL_PATH = "output/model-output/"
BASE_MODEL_ID = "google/gemma-3-1b-it"
LOCAL_ADAPTER_PATH = "./fine_tuned_adapter"

# Authenticate with Hugging Face
try:
    logger.info("Fetching HF Token from Secret Manager...")
    hf_token = access_secret_version(PROJECT_ID, SECRET_ID)
    logger.info("Logging into Hugging Face...")
    login(token=hf_token)
    logger.info("✅ Successfully logged into Hugging Face.")
except Exception as e:
    logger.error(f"❌ Failed to get secret or log in. Error: {e}")
    raise

# ==============================================================================
# PART 2: GCS & MODEL LOADING FUNCTIONS
# ==============================================================================

def download_gcs_folder(bucket_name, source_folder, destination_folder):
    """Downloads a folder from GCS."""
    try:
        os.makedirs(destination_folder, exist_ok=True)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=source_folder)
        
        logger.info(f"Downloading from gs://{bucket_name}/{source_folder}...")
        file_count = 0
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip folders
                destination_path = os.path.join(
                    destination_folder, 
                    os.path.relpath(blob.name, source_folder)
                )
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                blob.download_to_filename(destination_path)
                logger.info(f"Downloaded {blob.name} to {destination_path}")
                file_count += 1
        
        logger.info(f"✅ Downloaded {file_count} files from GCS")
        return file_count > 0
        
    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        raise

def load_peft_model(base_model_id, adapter_path):
    """Loads the base model and applies the PEFT adapter."""
    try:
        logger.info(f"Loading base model: {base_model_id}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply PEFT adapter
        logger.info(f"Applying PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        logger.info("✅ Model and adapter loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# ==============================================================================
# PART 3: FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Iris Flower Classification API",
    description="Fine-tuned Gemma model for Iris species prediction",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, tokenizer, model_loaded
    
    try:
        logger.info("Starting model loading process...")
        
        # Download model from GCS
        logger.info("Downloading fine-tuned adapter from GCS...")
        success = download_gcs_folder(BUCKET_NAME, GCS_MODEL_PATH, LOCAL_ADAPTER_PATH)
        
        if not success:
            raise Exception("No model files downloaded from GCS")
        
        # Load model and tokenizer
        logger.info("Loading base model and applying adapter...")
        model, tokenizer = load_peft_model(BASE_MODEL_ID, LOCAL_ADAPTER_PATH)
        model.eval()
        
        model_loaded = True
        logger.info("✅ Model loaded and ready for predictions")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model during startup: {e}")
        model_loaded = False
        raise

# Request/Response models
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class IrisResponse(BaseModel):
    prediction: str
    confidence: str = "high"
    processing_time_ms: float
    model_version: str = "1.0.0"

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Iris Flower Classification API",
        "model": BASE_MODEL_ID,
        "status": "ready" if model_loaded else "loading",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=IrisResponse)
async def predict(request: IrisRequest):
    """Predicts the Iris species based on flower measurements."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    start_time = time.time()
    
    try:
        # Format input using chat template
        user_content = (
            f"Sepal Length: {request.sepal_length}, Sepal Width: {request.sepal_width}, "
            f"Petal Length: {request.petal_length}, Petal Width: {request.petal_width}"
        )
        
        messages = [
            {
                "role": "system", 
                "content": "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"
            },
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Prediction made: {generated_text.strip()} (took {processing_time:.2f}ms)")
        
        return IrisResponse(
            prediction=generated_text.strip(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==============================================================================
# PART 4: ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )
