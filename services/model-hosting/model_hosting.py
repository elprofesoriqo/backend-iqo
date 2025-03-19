from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import os
import json
import asyncio
import httpx
import boto3
from botocore.exceptions import NoCredentialsError
import io
import docker
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor

# Models
class Model(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    model_type: str  # e.g., "classification", "segmentation", "nlp", etc.
    framework: str  # e.g., "pytorch", "tensorflow", "onnx", etc.
    version: str
    tags: List[str] = []
    file_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    is_public: bool = False
    metrics: Optional[Dict[str, Any]] = None
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ModelCreate(BaseModel):
    name: str
    description: str
    model_type: str
    framework: str
    version: str
    tags: List[str] = []
    is_public: bool = False
    metrics: Optional[Dict[str, Any]] = None

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    metrics: Optional[Dict[str, Any]] = None

class ModelSummary(BaseModel):
    id: str
    name: str
    description: str
    model_type: str
    framework: str
    version: str
    is_public: bool
    created_at: datetime

# Initialize FastAPI
app = FastAPI(title="Model Hosting Service API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME", "model_hosting"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        cursor_factory=RealDictCursor
    )

# S3 configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)
s3_bucket = os.environ.get("S3_BUCKET_NAME", "model-hosting-files")

# Docker client
docker_client = docker.from_env()

# User verification function
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('AUTH_SERVICE_URL', 'http://auth-service:8081')}/user",
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Function to validate model file format
def validate_model_format(file_path, framework):
    try:
        if framework == "onnx":
            import onnx
            onnx.load(file_path)
            return True
        elif framework == "pytorch":
            import torch
            torch.load(file_path, map_location=torch.device('cpu'))
            return True
        elif framework == "tensorflow":
            import tensorflow as tf
            tf.saved_model.load(file_path)
            return True
        return False
    except Exception as e:
        print(f"Model validation error: {e}")
        return False

# Function to optimize model
def optimize_model(file_path, framework):
    try:
        if framework == "onnx":
            # Use TensorRT for ONNX models
            optimized_path = file_path.replace(".onnx", "_optimized.onnx")
            subprocess.run(["trtexec", "--onnx=" + file_path, "--saveEngine=" + optimized_path])
            return optimized_path
        return file_path
    except Exception as e:
        print(f"Model optimization error: {e}")
        return file_path

# Function to package model into Docker container
def package_model_in_container(model_id, file_path, framework):
    try:
        container_name = f"model-{model_id}"
        # Create Dockerfile based on framework
        dockerfile_content = f"""
        FROM python:3.9-slim
        
        WORKDIR /app
        
        RUN pip install fastapi uvicorn {framework}
        
        COPY {os.path.basename(file_path)} /app/model
        COPY serve.py /app/
        
        EXPOSE 8000
        
        CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        
        # Create serving script
        serve_script = """
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import Dict, Any, List
        import json
        
        app = FastAPI()
        
        # Load model here based on framework
        # This is a placeholder
        class PredictionRequest(BaseModel):
            inputs: Any
            
        class PredictionResponse(BaseModel):
            outputs: Any
            
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            try:
                # Prediction logic goes here
                return {"outputs": request.inputs}  # Placeholder
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        """
        
        # Build and run Docker container
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        with open("serve.py", "w") as f:
            f.write(serve_script)
            
        docker_client.images.build(path=".", tag=container_name)
        container = docker_client.containers.run(
            container_name,
            detach=True,
            ports={"8000/tcp": None},  # Let Docker assign a port
            restart_policy={"Name": "always"}
        )
        
        # Get container port
        container_info = docker_client.api.inspect_container(container.id)
        host_port = list(container_info["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"])[0]
        
        return f"http://localhost:{host_port}"
    except Exception as e:
        print(f"Docker packaging error: {e}")
        return None

# Background task to process model
async def process_model_file(model_id, file_path, framework):
    try:
        # Validate format
        if not validate_model_format(file_path, framework):
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE models SET status = 'error', status_message = 'Invalid model format' WHERE id = %s",
                (model_id,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return
        
        # Optimize model
        optimized_path = optimize_model(file_path, framework)
        
        # Package in Docker container
        api_endpoint = package_model_in_container(model_id, optimized_path, framework)
        
        # Update database with API endpoint
        conn = get_db_connection()
        cursor = conn.cursor()
        if api_endpoint:
            cursor.execute(
                "UPDATE models SET status = 'deployed', api_endpoint = %s WHERE id = %s",
                (api_endpoint, model_id)
            )
        else:
            cursor.execute(
                "UPDATE models SET status = 'error', status_message = 'Failed to deploy container' WHERE id = %s",
                (model_id,)
            )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE models SET status = 'error', status_message = %s WHERE id = %s",
            (str(e), model_id)
        )
        conn.commit()
        cursor.close()
        conn.close()

# API endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/models", response_model=Model)
async def create_model(
    model: ModelCreate,
    current_user: dict = Depends(get_current_user)
):
    model_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO models (
                id, name, description, model_type, framework, version,
                tags, is_public, metrics, user_id, created_at, updated_at, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (
            model_id, model.name, model.description, model.model_type,
            model.framework, model.version, json.dumps(model.tags),
            model.is_public, json.dumps(model.metrics or {}),
            current_user["id"], now, now, "pending"
        ))
        
        new_model = dict(cursor.fetchone())
        conn.commit()
        
        return Model(**new_model)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/models", response_model=List[ModelSummary])
async def get_models(
    skip: int = 0,
    limit: int = 10,
    framework: Optional[str] = None,
    model_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = "SELECT * FROM models WHERE user_id = %s"
        params = [current_user["id"]]
        
        if framework:
            query += " AND framework = %s"
            params.append(framework)
        
        if model_type:
            query += " AND model_type = %s"
            params.append(model_type)
        
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, skip])
        
        cursor.execute(query, params)
        models = cursor.fetchall()
        
        return [ModelSummary(**dict(model)) for model in models]
    finally:
        cursor.close()
        conn.close()

@app.get("/models/{model_id}", response_model=Model)
async def get_model(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_dict = dict(model)
        
        # Check permissions
        if model_dict["user_id"] != current_user["id"] and not model_dict["is_public"]:
            raise HTTPException(status_code=403, detail="Not authorized to access this model")
        
        return Model(**model_dict)
    finally:
        cursor.close()
        conn.close()

@app.put("/models/{model_id}", response_model=Model)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if model exists and belongs to user
        cursor.execute(
            "SELECT * FROM models WHERE id = %s AND user_id = %s",
            (model_id, current_user["id"])
        )
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found or not authorized")
        
        # Update only provided fields
        update_data = {k: v for k, v in model_update.dict().items() if v is not None}
        if not update_data:
            return Model(**dict(model))
        
        # Convert lists and dicts to JSON strings
        if "tags" in update_data:
            update_data["tags"] = json.dumps(update_data["tags"])
        if "metrics" in update_data:
            update_data["metrics"] = json.dumps(update_data["metrics"])
        
        update_data["updated_at"] = datetime.utcnow()
        
        # Build update query
        set_clause = ", ".join([f"{key} = %s" for key in update_data.keys()])
        params = list(update_data.values())
        params.append(model_id)
        
        cursor.execute(
            f"UPDATE models SET {set_clause} WHERE id = %s RETURNING *",
            params
        )
        
        updated_model = dict(cursor.fetchone())
        conn.commit()
        
        return Model(**updated_model)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if model exists and belongs to user
        cursor.execute(
            "SELECT * FROM models WHERE id = %s AND user_id = %s",
            (model_id, current_user["id"])
        )
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found or not authorized")
        
        model_dict = dict(model)
        
        # Stop and remove Docker container if deployed
        if model_dict.get("api_endpoint"):
            try:
                container_name = f"model-{model_id}"
                container = docker_client.containers.get(container_name)
                container.stop()
                container.remove()
                docker_client.images.remove(container_name)
            except Exception as e:
                print(f"Error removing container: {e}")
        
        # Delete model file from S3 if exists
        if model_dict.get("file_url"):
            try:
                file_key = model_dict["file_url"].split(f"{s3_bucket}/")[-1]
                s3_client.delete_object(Bucket=s3_bucket, Key=file_key)
            except Exception as e:
                print(f"Error deleting file from S3: {e}")
        
        # Delete model from database
        cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))
        conn.commit()
        
        return {"message": "Model deleted successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/models/{model_id}/upload")
async def upload_model_file(
    model_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if model exists and belongs to user
        cursor.execute(
            "SELECT * FROM models WHERE id = %s AND user_id = %s",
            (model_id, current_user["id"])
        )
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found or not authorized")
        
        model_dict = dict(model)
        
        # Save file to S3
        file_content = await file.read()
        file_extension = os.path.splitext(file.filename)[1]
        file_key = f"models/{model_id}/{uuid.uuid4()}{file_extension}"
        
        s3_client.upload_fileobj(
            io.BytesIO(file_content),
            s3_bucket,
            file_key,
            ExtraArgs={"ContentType": file.content_type}
        )
        
        file_url = f"https://{s3_bucket}.s3.amazonaws.com/{file_key}"
        
        # Save locally for processing
        local_path = f"/tmp/{model_id}{file_extension}"
        with open(local_path, "wb") as f:
            f.write(file_content)
        
        # Update file URL in database
        cursor.execute(
            "UPDATE models SET file_url = %s, status = 'processing', updated_at = %s WHERE id = %s",
            (file_url, datetime.utcnow(), model_id)
        )
        conn.commit()
        
        # Process model in background
        background_tasks.add_task(
            process_model_file,
            model_id,
            local_path,
            model_dict["framework"]
        )
        
        return {"message": "Model file uploaded successfully and processing started"}
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not available")
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error uploading model file: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/models/{model_id}/predict")
async def predict_with_model(
    model_id: str,
    inputs: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_dict = dict(model)
        
        # Check permissions
        if model_dict["user_id"] != current_user["id"] and not model_dict["is_public"]:
            raise HTTPException(status_code=403, detail="Not authorized to use this model")
        
        # Check if model is deployed
        if model_dict["status"] != "deployed" or not model_dict.get("api_endpoint"):
            raise HTTPException(status_code=400, detail="Model is not deployed yet")
        
        # Forward prediction request to model API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{model_dict['api_endpoint']}/predict",
                json={"inputs": inputs},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/models/{model_id}/status")
async def get_model_status(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT id, status, status_message, api_endpoint FROM models WHERE id = %s AND user_id = %s",
            (model_id, current_user["id"])
        )
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found or not authorized")
        
        return dict(model)
    finally:
        cursor.close()
        conn.close()

@app.get("/huggingface/models")
async def search_huggingface_models(
    query: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://huggingface.co/api/models",
                params={"search": query, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Hugging Face models: {str(e)}")

@app.post("/huggingface/import")
async def import_from_huggingface(
    model_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Get model info from Hugging Face
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://huggingface.co/api/models/{model_id}")
            response.raise_for_status()
            hf_model = response.json()
        
        # Create model in database
        model_create = ModelCreate(
            name=hf_model["modelId"],
            description=hf_model.get("description", "Imported from Hugging Face"),
            model_type=hf_model.get("pipeline_tag", "unknown"),
            framework="pytorch",  # Default to PyTorch
            version="1.0.0",
            tags=hf_model.get("tags", []),
            is_public=False
        )
        
        # Create the model
        model = await create_model(model_create, current_user)
        
        # Download and process the model in background
        background_tasks.add_task(
            download_huggingface_model,
            model.id,
            hf_model["modelId"],
            current_user["id"]
        )
        
        return {"message": f"Started importing model {hf_model['modelId']}", "model_id": model.id}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing model: {str(e)}")

async def download_huggingface_model(model_id, hf_model_id, user_id):
    try:
        # Download model from Hugging Face
        from transformers import AutoModel
        
        local_path = f"/tmp/{model_id}"
        os.makedirs(local_path, exist_ok=True)
        
        # Download the model
        model = AutoModel.from_pretrained(hf_model_id)
        model.save_pretrained(local_path)
        
        # Upload to S3
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_key = f"models/{model_id}/{file}"
                
                s3_client.upload_file(
                    file_path,
                    s3_bucket,
                    file_key
                )
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        file_url = f"https://{s3_bucket}.s3.amazonaws.com/models/{model_id}/"
        
        cursor.execute(
            "UPDATE models SET file_url = %s, status = 'imported', updated_at = %s WHERE id = %s",
            (file_url, datetime.utcnow(), model_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Process the model
        await process_model_file(model_id, local_path, "pytorch")
    except Exception as e:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE models SET status = 'error', status_message = %s WHERE id = %s",
            (str(e), model_id)
        )
        conn.commit()
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)