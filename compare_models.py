# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import os
import time
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from google.cloud import storage, secretmanager
from huggingface_hub import login
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROJECT_ID = "premium-cipher-462011-p3"
SECRET_ID = "HF_TOKEN"
BASE_MODEL_ID = "google/gemma-3-1b-it"
BUCKET_NAME = "mlops-course-premium-cipher-462011-p3-unique"
GCS_ADAPTER_PATH = "output/model-output/"
LOCAL_ADAPTER_PATH = "./fine_tuned_adapter_for_testing"

# ==============================================================================
# SECTION 2: AUTHENTICATION
# ==============================================================================

def access_secret_version(project_id, secret_id, version_id="latest"):
    """Fetches a secret from Google Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"❌ Error accessing secret: {e}")
        return None

# Authentication
try:
    print("Fetching Hugging Face Token from Google Secret Manager...")
    hf_token = access_secret_version(PROJECT_ID, SECRET_ID)
    if hf_token:
        print("Logging into Hugging Face...")
        login(token=hf_token)
        print("✅ Successfully logged into Hugging Face.")
    else:
        raise Exception("Failed to retrieve HF token")
except Exception as e:
    print(f"❌ Failed to authenticate: {e}")
    exit()

# ==============================================================================
# SECTION 3: PERFORMANCE EVALUATION CLASS
# ==============================================================================

class ModelPerformanceEvaluator:
    def __init__(self):
        self.results = {
            "base_model": {
                "predictions": [],
                "actuals": [],
                "inference_times": [],
                "raw_predictions": []
            },
            "fine_tuned_model": {
                "predictions": [],
                "actuals": [],
                "inference_times": [],
                "raw_predictions": []
            },
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "base_model_id": BASE_MODEL_ID,
                "test_samples": 0
            }
        }
    
    def add_prediction(self, actual, base_pred, tuned_pred, base_time, tuned_time, base_raw, tuned_raw):
        """Add prediction results for both models."""
        normalized_actual = actual.lower()
        normalized_base = self._normalize_prediction(base_pred)
        normalized_tuned = self._normalize_prediction(tuned_pred)
        
        # Store results for base model
        self.results["base_model"]["predictions"].append(normalized_base)
        self.results["base_model"]["actuals"].append(normalized_actual)
        self.results["base_model"]["inference_times"].append(base_time)
        self.results["base_model"]["raw_predictions"].append(base_raw)
        
        # Store results for fine-tuned model
        self.results["fine_tuned_model"]["predictions"].append(normalized_tuned)
        self.results["fine_tuned_model"]["actuals"].append(normalized_actual)
        self.results["fine_tuned_model"]["inference_times"].append(tuned_time)
        self.results["fine_tuned_model"]["raw_predictions"].append(tuned_raw)
        
        self.results["evaluation_metadata"]["test_samples"] += 1
    
    def _normalize_prediction(self, prediction):
        """Normalize prediction text to standard class names."""
        pred_lower = prediction.lower().strip()
        
        if 'setosa' in pred_lower:
            return 'setosa'
        elif 'versicolor' in pred_lower:
            return 'versicolor'
        elif 'virginica' in pred_lower:
            return 'virginica'
        else:
            return 'unknown'
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics for both models."""
        for model_name in ["base_model", "fine_tuned_model"]:
            predictions = self.results[model_name]["predictions"]
            actuals = self.results[model_name]["actuals"]
            times = self.results[model_name]["inference_times"]
            
            # Classification metrics
            accuracy = accuracy_score(actuals, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                actuals, predictions, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(actuals, predictions, labels=['setosa', 'versicolor', 'virginica'])
            
            # Performance metrics
            avg_inference_time = np.mean(times)
            std_inference_time = np.std(times)
            
            # Per-class metrics
            class_report = classification_report(
                actuals, predictions, 
                labels=['setosa', 'versicolor', 'virginica'],
                output_dict=True,
                zero_division=0
            )
            
            self.results[model_name]["metrics"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "avg_inference_time_ms": float(avg_inference_time * 1000),
                "std_inference_time_ms": float(std_inference_time * 1000),
                "classification_report": class_report,
                "total_predictions": len(predictions),
                "correct_predictions": int(np.sum(np.array(predictions) == np.array(actuals)))
            }
    
    def generate_governance_report(self):
        """Generate governance compliance report."""
        self.calculate_metrics()
        
        base_metrics = self.results["base_model"]["metrics"]
        tuned_metrics = self.results["fine_tuned_model"]["metrics"]
        
        # Governance flags
        governance_flags = []
        recommendations = []
        
        # Performance regression check
        if tuned_metrics["accuracy"] < base_metrics["accuracy"]:
            governance_flags.append("PERFORMANCE_REGRESSION")
            recommendations.append("Fine-tuned model shows lower accuracy than base model")
        
        # Minimum accuracy threshold
        min_accuracy_threshold = 0.7
        if tuned_metrics["accuracy"] < min_accuracy_threshold:
            governance_flags.append("LOW_ACCURACY")
            recommendations.append(f"Model accuracy ({tuned_metrics['accuracy']:.3f}) below threshold ({min_accuracy_threshold})")
        
        # Performance consistency check
        if tuned_metrics["std_inference_time_ms"] > 1000:  # >1 second std dev
            governance_flags.append("INCONSISTENT_PERFORMANCE")
            recommendations.append("High variance in inference times")
        
        # Bias detection (class imbalance in errors)
        cm = np.array(tuned_metrics["confusion_matrix"])
        class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
        if np.std(class_accuracies) > 0.2:
            governance_flags.append("POTENTIAL_BIAS")
            recommendations.append("Significant performance differences across classes")
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_comparison": {
                "base_model_accuracy": base_metrics["accuracy"],
                "fine_tuned_model_accuracy": tuned_metrics["accuracy"],
                "improvement": tuned_metrics["accuracy"] - base_metrics["accuracy"],
                "base_avg_time": base_metrics["avg_inference_time_ms"],
                "tuned_avg_time": tuned_metrics["avg_inference_time_ms"]
            },
            "governance_assessment": {
                "flags": governance_flags,
                "recommendations": recommendations,
                "compliance_score": max(0, 100 - len(governance_flags) * 20),  # Simple scoring
                "risk_level": "HIGH" if len(governance_flags) > 2 else "MEDIUM" if len(governance_flags) > 0 else "LOW"
            },
            "detailed_metrics": self.results
        }
    
    def save_results(self, filename="model_comparison_results.json"):
        """Save results to file and upload to GCS."""
        report = self.generate_governance_report()
        
        # Save locally
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Upload to GCS
        try:
            storage_client = storage.Client(project=PROJECT_ID)
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(f"logs/governance/{filename}")
            blob.upload_from_filename(filename)
            print(f"✅ Results uploaded to gs://{BUCKET_NAME}/logs/governance/{filename}")
        except Exception as e:
            print(f"❌ Failed to upload to GCS: {e}")
        
        return report

# ==============================================================================
# SECTION 4: DATA PREPARATION
# ==============================================================================

print("\nPreparing sample data...")
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Select balanced sample for testing
sample_rows = pd.concat([
    iris_df[iris_df['species_name'] == 'setosa'].iloc[0:5],
    iris_df[iris_df['species_name'] == 'versicolor'].iloc[0:5],
    iris_df[iris_df['species_name'] == 'virginica'].iloc[0:5]
])

print("✅ Sample data is ready.")
print(f"Testing with {len(sample_rows)} samples")

# ==============================================================================
# SECTION 5: MODEL LOADING
# ==============================================================================

def download_gcs_folder(bucket_name, source_folder, destination_folder):
    """Downloads a folder and its contents from GCS."""
    os.makedirs(destination_folder, exist_ok=True)
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)
    
    print(f"\nDownloading adapter from gs://{bucket_name}/{source_folder}...")
    file_count = 0
    for blob in blobs:
        if not blob.name.endswith('/'):
            destination_path = os.path.join(destination_folder, os.path.relpath(blob.name, source_folder))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            file_count += 1
    
    print(f"✅ Downloaded {file_count} files")
    return file_count > 0

# Load base model
print("\nLoading original base model from Hugging Face...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✅ Base model loaded.")

# Download and load fine-tuned model
if download_gcs_folder(BUCKET_NAME, GCS_ADAPTER_PATH, LOCAL_ADAPTER_PATH):
    print("\nLoading fine-tuned model...")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, LOCAL_ADAPTER_PATH)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print("✅ Fine-tuned model loaded.")
else:
    print("❌ Failed to download model adapter")
    exit()

# ==============================================================================
# SECTION 6: PREDICTION AND COMPARISON
# ==============================================================================

def predict_species_with_timing(model, tokenizer, row):
    """Generate prediction with timing measurement."""
    start_time = time.time()
    
    user_content = (
        f"Sepal Length: {row['sepal length (cm)']}, Sepal Width: {row['sepal width (cm)']}, "
        f"Petal Length: {row['petal length (cm)']}, Petal Width: {row['petal width (cm)']}"
    )
    
    messages = [
        {"role": "system", "content": "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    inference_time = time.time() - start_time
    return generated_text.strip(), inference_time

# Initialize evaluator
evaluator = ModelPerformanceEvaluator()

print("\n" + "="*70)
print("STARTING COMPREHENSIVE MODEL COMPARISON")
print("="*70)

# Run predictions and collect results
for index, row in sample_rows.iterrows():
    print(f"\n🌸 Sample {index + 1}: {row['species_name'].capitalize()}")
    print(f"   Measurements: SL={row['sepal length (cm)']:.1f}, SW={row['sepal width (cm)']:.1f}, "
          f"PL={row['petal length (cm)']:.1f}, PW={row['petal width (cm)']:.1f}")
    
    # Get predictions with timing
    base_prediction, base_time = predict_species_with_timing(base_model, base_tokenizer, row)
    tuned_prediction, tuned_time = predict_species_with_timing(fine_tuned_model, fine_tuned_tokenizer, row)
    
    print(f"   🧠 Base Model: '{base_prediction}' ({base_time*1000:.1f}ms)")
    print(f"   ✨ Fine-Tuned: '{tuned_prediction}' ({tuned_time*1000:.1f}ms)")
    
    # Add to evaluator
    evaluator.add_prediction(
        actual=row['species_name'],
        base_pred=base_prediction,
        tuned_pred=tuned_prediction,
        base_time=base_time,
        tuned_time=tuned_time,
        base_raw=base_prediction,
        tuned_raw=tuned_prediction
    )

# Generate and save comprehensive report
print("\n" + "="*70)
print("GENERATING GOVERNANCE REPORT")
print("="*70)

report = evaluator.save_results()

# Print summary
print(f"\n📊 PERFORMANCE SUMMARY:")
print(f"   Base Model Accuracy: {report['model_comparison']['base_model_accuracy']:.3f}")
print(f"   Fine-Tuned Accuracy: {report['model_comparison']['fine_tuned_model_accuracy']:.3f}")
print(f"   Improvement: {report['model_comparison']['improvement']:+.3f}")
print(f"   Base Avg Time: {report['model_comparison']['base_avg_time']:.1f}ms")
print(f"   Tuned Avg Time: {report['model_comparison']['tuned_avg_time']:.1f}ms")

print(f"\n🏛️ GOVERNANCE ASSESSMENT:")
print(f"   Compliance Score: {report['governance_assessment']['compliance_score']}/100")
print(f"   Risk Level: {report['governance_assessment']['risk_level']}")

if report['governance_assessment']['flags']:
    print(f"   ⚠️ Flags: {', '.join(report['governance_assessment']['flags'])}")
    for rec in report['governance_assessment']['recommendations']:
        print(f"   💡 {rec}")
else:
    print("   ✅ No governance violations detected")

print(f"\n✅ Complete report saved and uploaded to GCS")
