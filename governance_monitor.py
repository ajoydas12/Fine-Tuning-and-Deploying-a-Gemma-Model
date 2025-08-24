# ==============================================================================
# GOVERNANCE MONITORING AND COMPLIANCE SYSTEM
# ==============================================================================

import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import os
from google.cloud import storage, monitoring_v3
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ==============================================================================

@dataclass
class GovernanceConfig:
    """Configuration for governance parameters."""
    performance_thresholds: Dict[str, float]
    security_requirements: Dict[str, bool]
    compliance_checks: Dict[str, bool]
    monitoring_intervals: Dict[str, int]
    audit_retention_days: int = 90

# Default governance configuration
DEFAULT_GOVERNANCE_CONFIG = GovernanceConfig(
    performance_thresholds={
        "minimum_accuracy": 0.80,
        "maximum_inference_time_ms": 2000.0,
        "maximum_error_rate": 0.05,
        "minimum_throughput_rps": 1.0,
        "maximum_memory_usage_mb": 8192,
        "maximum_cpu_usage_percent": 80.0
    },
    security_requirements={
        "encrypt_model_artifacts": True,
        "secure_api_authentication": True,
        "audit_logging_enabled": True,
        "data_privacy_compliance": True,
        "access_control_enabled": True
    },
    compliance_checks={
        "model_versioning_required": True,
        "data_lineage_tracking": True,
        "performance_monitoring": True,
        "bias_detection": True,
        "explainability_required": False,
        "drift_detection": True
    },
    monitoring_intervals={
        "performance_check_seconds": 300,
        "drift_detection_seconds": 3600,
        "compliance_audit_seconds": 86400,
        "health_check_seconds": 60
    }
)

# ==============================================================================
# GOVERNANCE MONITOR CLASS
# ==============================================================================

class GovernanceMonitor:
    def __init__(self, config: GovernanceConfig = DEFAULT_GOVERNANCE_CONFIG, 
                 project_id: str = "premium-cipher-462011-p3"):
        self.config = config
        self.project_id = project_id
        self.audit_log: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.compliance_history: List[Dict[str, Any]] = []
        
        # Initialize GCS client for audit logging
        try:
            self.storage_client = storage.Client(project=project_id)
            logger.info("✅ GCS client initialized for audit logging")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCS client: {e}")
            self.storage_client = None
    
    def log_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log governance events with timestamps."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "event_id": hashlib.sha256(
                f"{datetime.now().isoformat()}{event_type}{json.dumps(details, sort_keys=True)}".encode()
            ).hexdigest()[:16]
        }
        
        self.audit_log.append(log_entry)
        logger.info(f"[{severity}] {event_type}: {details}")
        
        # Upload to GCS if available
        if self.storage_client:
            try:
                self._upload_audit_log()
            except Exception as e:
                logger.error(f"Failed to upload audit log: {e}")
    
    def check_model_governance(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model governance compliance check."""
        compliance_report = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_info.get("model_id", "unknown"),
            "model_version": model_info.get("version", "unknown"),
            "compliance_checks": {},
            "violations": [],
            "warnings": [],
            "recommendations": [],
            "compliance_score": 0,
            "risk_assessment": {}
        }
        
        violations = []
        warnings = []
        recommendations = []
        
        # 1. Model Versioning Check
        if self.config.compliance_checks["model_versioning_required"]:
            if not model_info.get("version"):
                violations.append("Missing model version identifier")
                recommendations.append("Implement semantic versioning for model releases")
            elif not self._validate_version_format(model_info.get("version", "")):
                warnings.append("Model version doesn't follow semantic versioning")
                recommendations.append("Use semantic versioning format (e.g., 1.2.3)")
        
        # 2. Performance Thresholds
        accuracy = model_info.get("accuracy", 0)
        if accuracy < self.config.performance_thresholds["minimum_accuracy"]:
            violations.append(f"Model accuracy ({accuracy:.3f}) below threshold ({self.config.performance_thresholds['minimum_accuracy']})")
            recommendations.append("Retrain model or adjust acceptance criteria")
        
        inference_time = model_info.get("avg_inference_time_ms", 0)
        if inference_time > self.config.performance_thresholds["maximum_inference_time_ms"]:
            violations.append(f"Average inference time ({inference_time:.1f}ms) exceeds threshold")
            recommendations.append("Optimize model or infrastructure for better performance")
        
        # 3. Data Lineage Check
        if self.config.compliance_checks["data_lineage_tracking"]:
            if not model_info.get("training_data_hash"):
                violations.append("Missing training data lineage tracking")
                recommendations.append("Implement data versioning and hash tracking")
            
            if not model_info.get("training_timestamp"):
                warnings.append("Missing training timestamp for audit trail")
                recommendations.append("Record training timestamps for full traceability")
        
        # 4. Security Compliance
        if self.config.security_requirements["encrypt_model_artifacts"]:
            if not model_info.get("encryption_enabled", False):
                violations.append("Model artifacts are not encrypted")
                recommendations.append("Enable encryption for model storage and transit")
        
        if self.config.security_requirements["secure_api_authentication"]:
            if not model_info.get("authentication_enabled", False):
                violations.append("API authentication not enabled")
                recommendations.append("Implement proper API authentication and authorization")
        
        # 5. Bias and Fairness Check
        if self.config.compliance_checks["bias_detection"]:
            bias_metrics = model_info.get("bias_metrics", {})
            if not bias_metrics:
                warnings.append("No bias detection metrics available")
                recommendations.append("Implement bias detection and fairness metrics")
            else:
                # Check for significant bias across classes
                class_accuracies = bias_metrics.get("class_accuracies", [])
                if class_accuracies and np.std(class_accuracies) > 0.15:
                    warnings.append("Potential bias detected across different classes")
                    recommendations.append("Review training data balance and model fairness")
        
        # 6. Resource Usage Check
        memory_usage = model_info.get("memory_usage_mb", 0)
        if memory_usage > self.config.performance_thresholds["maximum_memory_usage_mb"]:
            warnings.append(f"High memory usage detected: {memory_usage}MB")
            recommendations.append("Consider model compression or infrastructure scaling")
        
        # Calculate compliance score
        total_checks = len(self.config.compliance_checks) + len(self.config.security_requirements)
        violations_count = len(violations)
        warnings_count = len(warnings)
        
        compliance_score = max(0, 100 - (violations_count * 20) - (warnings_count * 5))
        
        # Risk assessment
        risk_level = "LOW"
        if violations_count >= 3 or compliance_score < 60:
            risk_level = "HIGH"
        elif violations_count >= 1 or warnings_count >= 3 or compliance_score < 80:
            risk_level = "MEDIUM"
        
        # Update report
        compliance_report.update({
            "violations": violations,
            "warnings": warnings,
            "recommendations": recommendations,
            "compliance_score": compliance_score,
            "risk_assessment": {
                "risk_level": risk_level,
                "violations_count": violations_count,
                "warnings_count": warnings_count,
                "total_checks_performed": total_checks
            }
        })
        
        # Log the governance check
        self.log_event("governance_check", {
            "model_id": model_info.get("model_id"),
            "compliance_score": compliance_score,
            "risk_level": risk_level,
            "violations_count": violations_count
        }, severity="WARNING" if violations else "INFO")
        
        self.compliance_history.append(compliance_report)
        return compliance_report
    
    def monitor_model_performance(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor ongoing model performance against governance thresholds."""
        monitoring_report = {
            "timestamp": datetime.now().isoformat(),
            "performance_status": "HEALTHY",
            "alerts": [],
            "metrics": performance_metrics,
            "threshold_violations": []
        }
        
        alerts = []
        threshold_violations = []
        
        # Check accuracy degradation
        current_accuracy = performance_metrics.get("accuracy", 0)
        if current_accuracy < self.config.performance_thresholds["minimum_accuracy"]:
            alerts.append(f"Accuracy below threshold: {current_accuracy:.3f}")
            threshold_violations.append("minimum_accuracy")
        
        # Check inference time
        avg_inference_time = performance_metrics.get("avg_inference_time_ms", 0)
        if avg_inference_time > self.config.performance_thresholds["maximum_inference_time_ms"]:
            alerts.append(f"Inference time exceeded: {avg_inference_time:.1f}ms")
            threshold_violations.append("maximum_inference_time_ms")
        
        # Check error rate
        error_rate = performance_metrics.get("error_rate", 0)
        if error_rate > self.config.performance_thresholds["maximum_error_rate"]:
            alerts.append(f"Error rate too high: {error_rate:.3f}")
            threshold_violations.append("maximum_error_rate")
        
        # Determine overall status
        if threshold_violations:
            monitoring_report["performance_status"] = "DEGRADED" if len(threshold_violations) <= 2 else "CRITICAL"
        
        monitoring_report["alerts"] = alerts
        monitoring_report["threshold_violations"] = threshold_violations
        
        # Log performance monitoring
        self.log_event("performance_monitoring", {
            "status": monitoring_report["performance_status"],
            "alerts_count": len(alerts),
            "violations": threshold_violations
        }, severity="WARNING" if alerts else "INFO")
        
        self.metrics_history.append(monitoring_report)
        return monitoring_report
    
    def detect_data_drift(self, current_predictions: List[str], 
                         baseline_predictions: List[str], 
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Detect data drift using prediction distributions."""
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_score": 0.0,
            "drift_analysis": {},
            "recommendation": ""
        }
        
        if len(current_predictions) < 10 or len(baseline_predictions) < 10:
            drift_report["recommendation"] = "Insufficient data for drift detection"
            return drift_report
        
        try:
            # Convert predictions to class distributions
            current_dist = self._calculate_class_distribution(current_predictions)
            baseline_dist = self._calculate_class_distribution(baseline_predictions)
            
            # Calculate KL divergence as drift score
            drift_score = self._calculate_kl_divergence(baseline_dist, current_dist)
            
            drift_detected = drift_score > threshold
            
            drift_report.update({
                "drift_detected": drift_detected,
                "drift_score": float(drift_score),
                "drift_analysis": {
                    "current_distribution": current_dist,
                    "baseline_distribution": baseline_dist,
                    "threshold": threshold
                },
                "recommendation": "Model retraining recommended" if drift_detected else "No action needed"
            })
            
            # Log drift detection
            self.log_event("drift_detection", {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "threshold": threshold
            }, severity="WARNING" if drift_detected else "INFO")
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            drift_report["recommendation"] = f"Drift detection failed: {str(e)}"
        
        return drift_report
    
    def generate_compliance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Generate compliance summary for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent compliance checks
        recent_checks = [
            check for check in self.compliance_history
            if datetime.fromisoformat(check["timestamp"]) > cutoff_date
        ]
        
        if not recent_checks:
            return {"message": f"No compliance checks in the last {days} days"}
        
        # Calculate summary statistics
        avg_compliance_score = np.mean([check["compliance_score"] for check in recent_checks])
        total_violations = sum(len(check["violations"]) for check in recent_checks)
        total_warnings = sum(len(check["warnings"]) for check in recent_checks)
        
        risk_levels = [check["risk_assessment"]["risk_level"] for check in recent_checks]
        risk_distribution = {level: risk_levels.count(level) for level in ["LOW", "MEDIUM", "HIGH"]}
        
        return {
            "period_days": days,
            "total_checks": len(recent_checks),
            "average_compliance_score": round(avg_compliance_score, 2),
            "total_violations": total_violations,
            "total_warnings": total_warnings,
            "risk_distribution": risk_distribution,
            "trend": self._calculate_compliance_trend(recent_checks),
            "generated_at": datetime.now().isoformat()
        }
    
    # Helper Methods
    def _validate_version_format(self, version: str) -> bool:
        """Validate semantic versioning format."""
        import re
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        return bool(re.match(pattern, version))
    
    def _calculate_class_distribution(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate class distribution from predictions."""
        total = len(predictions)
        if total == 0:
            return {}
        
        class_counts = {}
        for pred in predictions:
            pred_normalized = pred.lower().strip()
            class_counts[pred_normalized] = class_counts.get(pred_normalized, 0) + 1
        
        return {cls: count / total for cls, count in class_counts.items()}
    
    def _calculate_kl_divergence(self, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        """Calculate KL divergence between two distributions."""
        all_classes = set(p_dist.keys()) | set(q_dist.keys())
        kl_div = 0.0
        
        for cls in all_classes:
            p = p_dist.get(cls, 1e-8)  # Small epsilon to avoid log(0)
            q = q_dist.get(cls, 1e-8)
            kl_div += p * np.log(p / q)
        
        return kl_div
    
    def _calculate_compliance_trend(self, recent_checks: List[Dict]) -> str:
        """Calculate compliance trend over time."""
        if len(recent_checks) < 2:
            return "INSUFFICIENT_DATA"
        
        scores = [check["compliance_score"] for check in recent_checks]
        if scores[-1] > scores[0]:
            return "IMPROVING"
        elif scores[-1] < scores[0]:
            return "DECLINING"
        else:
            return "STABLE"
    
    def _upload_audit_log(self):
        """Upload audit log to GCS."""
        if not self.storage_client or not self.audit_log:
            return
        
        try:
            bucket_name = "mlops-course-premium-cipher-462011-p3-unique"
            bucket = self.storage_client.bucket(bucket_name)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/governance/audit_log_{timestamp}.json"
            
            # Upload recent audit entries
            recent_logs = self.audit_log[-100:]  # Keep last 100 entries
            blob = bucket.blob(filename)
            blob.upload_from_string(json.dumps(recent_logs, indent=2))
            
        except Exception as e:
            logger.error(f"Failed to upload audit log: {e}")

# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Initialize governance monitor
    monitor = GovernanceMonitor()
    
    # Example model info for governance check
    model_info = {
        "model_id": "iris-classifier-gemma",
        "version": "1.2.3",
        "accuracy": 0.85,
        "avg_inference_time_ms": 150.0,
        "training_data_hash": "abc123def456",
        "training_timestamp": "2025-08-24T10:00:00Z",
        "encryption_enabled": True,
        "authentication_enabled": False,
        "memory_usage_mb": 2048,
        "bias_metrics": {
            "class_accuracies": [0.9, 0.85, 0.82]
        }
    }
    
    # Run governance check
    print("Running governance compliance check...")
    compliance_report = monitor.check_model_governance(model_info)
    
    print(f"Compliance Score: {compliance_report['compliance_score']}/100")
    print(f"Risk Level: {compliance_report['risk_assessment']['risk_level']}")
    
    if compliance_report['violations']:
        print("\nViolations:")
        for violation in compliance_report['violations']:
            print(f"  ❌ {violation}")
    
    if compliance_report['warnings']:
        print("\nWarnings:")
        for warning in compliance_report['warnings']:
            print(f"  ⚠️ {warning}")
    
    if compliance_report['recommendations']:
        print("\nRecommendations:")
        for rec in compliance_report['recommendations']:
            print(f"  💡 {rec}")
