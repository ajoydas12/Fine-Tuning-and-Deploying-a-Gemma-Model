# Updated README.md with GCS Commands

Here's the updated README.md with comprehensive GCS commands for the MLOps project:

```markdown
# MLOps Project: Fine-Tuning and Deploying a Gemma Model

This project fine-tunes a Gemma model on the Iris dataset and deploys it as a prediction API using FastAPI on a Google Cloud Vertex AI Workbench instance.

---

## Prerequisites

### Google Cloud Setup
```
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set your project
gcloud config set project premium-cipher-462011-p3

# Authenticate
gcloud auth login
gcloud auth application-default login

# Enable required APIs
gcloud services enable storage-component.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable notebooks.googleapis.com
```

---

## Phase 1: Setup and Data Preparation

### 1. Create GCS Resources
```
# Set environment variables
export PROJECT_ID="premium-cipher-462011-p3"
export BUCKET_NAME="mlops-course-premium-cipher-462011-p3-unique"
export REGION="us-central1"

# Create GCS bucket
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME

# Create folder structure in GCS
gsutil -m cp -r /dev/null gs://$BUCKET_NAME/input/
gsutil -m cp -r /dev/null gs://$BUCKET_NAME/output/
gsutil -m cp -r /dev/null gs://$BUCKET_NAME/models/
gsutil -m cp -r /dev/null gs://$BUCKET_NAME/logs/
```

### 2. Setup Secret Manager
```
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create secret for Hugging Face token
echo -n "YOUR_HF_TOKEN_HERE" | gcloud secrets create HF_TOKEN --data-file=-

# Create secret for GCP service account (for Kaggle)
gcloud iam service-accounts create mlops-service-account --display-name="MLOps Service Account"

# Generate and download service account key
gcloud iam service-accounts keys create gcp-key.json \
    --iam-account=mlops-service-account@$PROJECT_ID.iam.gserviceaccount.com

# Store the service account key in Secret Manager
gcloud secrets create GCP_USER_CREDENTIALS --data-file=gcp-key.json

# Grant necessary permissions to service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:mlops-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:mlops-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 3. Prepare and Upload Training Data
```
# Create local data directory
mkdir -p ./data

# Generate training data (run this Python script locally first)
python generate_iris_data.py

# Upload training data to GCS
gsutil -m cp ./data/*.jsonl gs://$BUCKET_NAME/input/

# Verify upload
gsutil ls -la gs://$BUCKET_NAME/input/
```

---

## Phase 2: Model Fine-Tuning (Kaggle)

### 1. Download Data in Kaggle Notebook
```
# Add this to your Kaggle notebook
import os
from google.cloud import storage

# Download training data from GCS
def download_training_data():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Download train.jsonl
    blob = bucket.blob('input/train.jsonl')
    blob.download_to_filename('./train.jsonl')
    
    # Download eval.jsonl
    blob = bucket.blob('input/eval.jsonl')
    blob.download_to_filename('./eval.jsonl')
    
    print("✅ Training data downloaded from GCS")

download_training_data()
```

### 2. Upload Model After Training
```
# Add this to your Kaggle notebook after training
def upload_model_to_gcs(local_path="./output"):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    import os
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            gcs_path = f"output/model-output/{relative_path}"
            
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{BUCKET_NAME}/{gcs_path}")
    
    print("✅ Model uploaded to GCS")

upload_model_to_gcs()
```

---

## Phase 3: Vertex AI Workbench Setup

### 1. Create Vertex AI Workbench Instance
```
# Create Vertex AI Workbench instance
gcloud notebooks instances create mlops-workbench \
    --location=$REGION \
    --machine-type=n1-standard-4 \
    --accelerator-type=NVIDIA_TESLA_T4 \
    --accelerator-core-count=1 \
    --boot-disk-type=PD_SSD \
    --boot-disk-size=100GB \
    --data-disk-type=PD_SSD \
    --data-disk-size=200GB \
    --install-gpu-driver \
    --async

# Check instance status
gcloud notebooks instances list --location=$REGION

# Get instance details
gcloud notebooks instances describe mlops-workbench --location=$REGION
```

### 2. Connect to Workbench Instance
```
# SSH into the instance
gcloud compute ssh mlops-workbench --zone=$REGION-a

# Or use the web interface
gcloud notebooks instances get-health mlops-workbench --location=$REGION
```

### 3. Setup Project Environment on Workbench
```
# On the Workbench instance, create project directory
mkdir -p /home/jupyter/mlops-project
cd /home/jupyter/mlops-project

# Clone or download your project files from GCS
gsutil -m cp gs://$BUCKET_NAME/code/* ./

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Grant IAM Permissions for Workbench
```
# Get the service account used by your Workbench instance
WORKBENCH_SA=$(gcloud notebooks instances describe mlops-workbench \
    --location=$REGION \
    --format="value(serviceAccount)")

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$WORKBENCH_SA" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$WORKBENCH_SA" \
    --role="roles/secretmanager.secretAccessor"
```

---

## Phase 4: Model Deployment and Testing

### 1. Download Model from GCS
```
# On Workbench instance, download the fine-tuned model
cd /home/jupyter/mlops-project

# Download model artifacts
gsutil -m cp -r gs://$BUCKET_NAME/output/model-output/ ./fine_tuned_adapter/

# Verify download
ls -la ./fine_tuned_adapter/
```

### 2. Run Model Comparison
```
# Activate environment
source .venv/bin/activate

# Run model comparison
python compare_models.py

# Upload comparison results to GCS
gsutil cp comparison_results.json gs://$BUCKET_NAME/logs/
```

### 3. Deploy API Server
```
# Start the FastAPI server
screen -S fastapi-server
cd /home/jupyter/mlops-project
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Detach from screen session
# Press Ctrl+A then D
```

### 4. Configure Firewall for External Access
```
# Create firewall rule for API access
gcloud compute firewall-rules create allow-fastapi \
    --allow tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow FastAPI on port 8000"

# Get external IP of your Workbench instance
EXTERNAL_IP=$(gcloud compute instances describe mlops-workbench \
    --zone=$REGION-a \
    --format="get(networkInterfaces.accessConfigs.natIP)")

echo "API available at: http://$EXTERNAL_IP:8000"
```

---

## Phase 5: Testing and Monitoring

### 1. Test API Endpoints
```
# Test prediction endpoint
curl -X POST "http://$EXTERNAL_IP:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Test governance metrics endpoint (if implemented)
curl -X GET "http://$EXTERNAL_IP:8000/governance/metrics"

# Test health endpoint
curl -X GET "http://$EXTERNAL_IP:8000/docs"
```

### 2. Monitor Logs and Performance
```
# Monitor API logs
screen -r fastapi-server

# Upload logs to GCS periodically
gsutil cp /home/jupyter/mlops-project/logs/* gs://$BUCKET_NAME/logs/

# Monitor system resources
htop
nvidia-smi  # If using GPU
```

### 3. Backup and Version Control
```
# Upload all project files to GCS for backup
gsutil -m cp -r /home/jupyter/mlops-project gs://$BUCKET_NAME/backups/$(date +%Y%m%d)/

# Create a snapshot of the model
gsutil -m cp -r ./fine_tuned_adapter gs://$BUCKET_NAME/models/v1.0/
```

---

## GCS Management Commands

### Useful GCS Operations
```
# List all objects in bucket
gsutil ls -r gs://$BUCKET_NAME

# Check bucket storage usage
gsutil du -sh gs://$BUCKET_NAME

# Copy files between GCS locations
gsutil -m cp -r gs://$BUCKET_NAME/output/model-output/ gs://$BUCKET_NAME/models/production/

# Delete old files
gsutil -m rm gs://$BUCKET_NAME/logs/old-logs/*

# Set object lifecycle policies
gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME

# Make objects public (if needed)
gsutil -m acl ch -u AllUsers:R gs://$BUCKET_NAME/models/production/*
```

### Cleanup Commands
```
# Delete Workbench instance
gcloud notebooks instances delete mlops-workbench --location=$REGION

# Delete firewall rule
gcloud compute firewall-rules delete allow-fastapi

# Delete service account
gcloud iam service-accounts delete mlops-service-account@$PROJECT_ID.iam.gserviceaccount.com

# Delete secrets
gcloud secrets delete HF_TOKEN
gcloud secrets delete GCP_USER_CREDENTIALS

# Delete GCS bucket (WARNING: This deletes all data)
gsutil -m rm -r gs://$BUCKET_NAME
```

---

## Environment Variables File

Create a `.env` file for consistent configuration:
```
# .env file
export PROJECT_ID="premium-cipher-462011-p3"
export BUCKET_NAME="mlops-course-premium-cipher-462011-p3-unique"
export REGION="us-central1"
export BASE_MODEL_ID="google/gemma-3-1b-it"
export SECRET_ID="HF_TOKEN"
export GCS_ADAPTER_PATH="output/model-output/"

# Source this file
source .env
```

---

## Troubleshooting

### Common Issues and Solutions
```
# Authentication issues
gcloud auth list
gcloud config list

# Permission issues
gcloud projects get-iam-policy $PROJECT_ID

# GCS access issues
gsutil ls gs://$BUCKET_NAME  # Test basic access

# Check service account permissions
gcloud iam service-accounts get-iam-policy mlops-service-account@$PROJECT_ID.iam.gserviceaccount.com

# Network connectivity
curl -I http://EXTERNAL_IP:8000/docs
```

This comprehensive README provides all the necessary GCS commands and setup instructions for the complete MLOps pipeline from data preparation to model deployment and monitoring.
```

This updated README includes:

1. **Comprehensive GCS setup commands** for buckets, folders, and permissions
2. **Secret Manager configuration** for secure credential storage
3. **Vertex AI Workbench creation and setup** commands
4. **Complete deployment pipeline** with GCS integration
5. **Monitoring and backup procedures** using GCS
6. **Troubleshooting section** for common issues
7. **Cleanup commands** for resource management
8. **Environment configuration** for consistent setup
