import requests
import json

def trigger_airflow_dag(config):
    """Trigger DAG using Airflow REST API"""
    url = "http://localhost:8080/api/v1/dags/mmo_evobagging_pipeline/dagRuns"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    payload = {
        "conf": config,
        "replace_microseconds": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error triggering DAG: {str(e)}")
        return None

# Trigger DAG with custom configuration
conf = {
    'dataset_name': 'pima',
    'test_size': 0.2,
    'n_exp': 1,
    'n_bags': 8,
    'n_iter': 1
}

trigger_airflow_dag(conf) 