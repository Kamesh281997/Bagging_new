import subprocess
import time
import os
import docker

def check_docker_running():
    """Check if Docker is running"""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except:
        return False

def start_services():
    """Start MLflow and Airflow services"""
    try:
        if not check_docker_running():
            return None, "Docker is not running. Please start Docker Desktop first."

        # Start MLflow
        mlflow_proc = subprocess.Popen(
            ["mlflow", "server", "--host", "127.0.0.1", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Start Airflow using Docker Compose
        try:
            subprocess.run(
                ["docker-compose", "up", "-d"], 
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            return mlflow_proc, "Services started successfully!"
        except subprocess.CalledProcessError as e:
            return None, f"Error starting Docker services: {str(e)}"
        
    except Exception as e:
        return None, f"Error starting services: {str(e)}"

def stop_services(mlflow_proc):
    """Stop all services"""
    try:
        # Stop MLflow
        if mlflow_proc:
            mlflow_proc.terminate()
            
        # Stop Airflow containers
        subprocess.run(
            ["docker-compose", "down"], 
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return "Services stopped successfully!"
        
    except Exception as e:
        return f"Error stopping services: {str(e)}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            proc, msg = start_services()
            print(msg)
        elif sys.argv[1] == "stop":
            msg = stop_services(None)
            print(msg)