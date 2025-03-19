from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import os
import json
import motor.motor_asyncio
import boto3
from botocore.exceptions import NoCredentialsError
import httpx
import plotly.express as px
import pandas as pd
import io
import asyncio

# Modele danych
class Publication(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = []
    file_url: Optional[str] = None
    summary: Optional[str] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PublicationCreate(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = []

class PublicationUpdate(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    keywords: Optional[List[str]] = None

class PublicationSummary(BaseModel):
    id: str
    title: str
    authors: List[str]
    publication_date: datetime
    journal: Optional[str] = None
    summary: Optional[str] = None

# Inicjalizacja aplikacji
app = FastAPI(title="Research Service API")

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji należy ograniczyć
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfiguracja uwierzytelniania
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Konfiguracja MongoDB
mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
db = mongodb_client.research_platform
publications_collection = db.publications

# Konfiguracja S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)
s3_bucket = os.environ.get("S3_BUCKET_NAME", "research-platform-files")

# Funkcja do weryfikacji tokenu JWT
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

# Funkcja do generowania podsumowań za pomocą GPT-4
async def generate_summary(text: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that summarizes scientific papers."},
                        {"role": "user", "content": f"Please provide a concise summary of the following scientific paper abstract in 3-4 sentences: {text}"}
                    ],
                    "max_tokens": 150
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# Funkcja do generowania wizualizacji
def generate_visualizations(data):
    try:
        # W rzeczywistej implementacji dane byłyby analizowane i generowane odpowiednie wizualizacje
        # Tutaj tylko przykład
        fig = px.bar(pd.DataFrame({"value": [1, 2, 3]}), y="value")
        return [{"type": "plotly", "data": fig.to_json()}]
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return []

# Funkcja do pobierania publikacji z Google Scholar
async def fetch_publications_from_google_scholar(query: str, limit: int = 10):
    # W rzeczywistej implementacji byłaby integracja z Google Scholar API
    # Tutaj tylko symulacja
    await asyncio.sleep(1)  # Symulacja opóźnienia odpowiedzi
    sample_publications = [
        {
            "title": f"Sample Publication about {query} #{i}",
            "authors": ["Author A", "Author B"],
            "abstract": f"This is a sample abstract about {query}.",
            "publication_date": datetime.utcnow().isoformat(),
            "journal": "Sample Journal",
            "doi": f"10.1000/sample-{i}",
            "url": f"https://example.com/paper/{i}"
        }
        for i in range(1, limit + 1)
    ]
    return sample_publications

# Endpointy API
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/publications", response_model=Publication)
async def create_publication(
    publication: PublicationCreate,
    current_user: dict = Depends(get_current_user)
):
    # Tworzenie nowej publikacji
    pub_dict = publication.dict()
    pub_id = str(uuid.uuid4())
    summary = await generate_summary(pub_dict["abstract"])
    
    new_publication = Publication(
        id=pub_id,
        **pub_dict,
        summary=summary,
        visualizations=generate_visualizations(pub_dict),
        user_id=current_user["id"]
    )
    
    await publications_collection.insert_one(new_publication.dict())
    return new_publication

@app.get("/publications", response_model=List[PublicationSummary])
async def get_publications(
    skip: int = 0,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    # Pobieranie publikacji
    cursor = publications_collection.find({"user_id": current_user["id"]})
    publications = await cursor.skip(skip).limit(limit).to_list(length=limit)
    return publications

@app.get("/publications/{publication_id}", response_model=Publication)
async def get_publication(
    publication_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Pobieranie konkretnej publikacji
    publication = await publications_collection.find_one({"id": publication_id})
    if publication is None:
        raise HTTPException(status_code=404, detail="Publication not found")
    
    # Sprawdzanie uprawnień
    if publication["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this publication")
        
    return publication

@app.put("/publications/{publication_id}", response_model=Publication)
async def update_publication(
    publication_id: str,
    publication_update: PublicationUpdate,
    current_user: dict = Depends(get_current_user)
):
    # Pobieranie publikacji
    publication = await publications_collection.find_one({"id": publication_id})
    if publication is None:
        raise HTTPException(status_code=404, detail="Publication not found")
    
    # Sprawdzanie uprawnień
    if publication["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to update this publication")
    
    # Aktualizacja publikacji
    update_data = {k: v for k, v in publication_update.dict().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()
    
    if "abstract" in update_data:
        # Regeneracja podsumowania, jeśli abstrakt się zmienił
        update_data["summary"] = await generate_summary(update_data["abstract"])
    
    await publications_collection.update_one(
        {"id": publication_id},
        {"$set": update_data}
    )
    
    updated_publication = await publications_collection.find_one({"id": publication_id})
    return updated_publication

@app.delete("/publications/{publication_id}")
async def delete_publication(
    publication_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Pobieranie publikacji
    publication = await publications_collection.find_one({"id": publication_id})
    if publication is None:
        raise HTTPException(status_code=404, detail="Publication not found")
    
    # Sprawdzanie uprawnień
    if publication["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this publication")
    
    # Usuwanie pliku z S3, jeśli istnieje
    if publication.get("file_url"):
        try:
            file_key = publication["file_url"].split(f"{s3_bucket}/")[-1]
            s3_client.delete_object(Bucket=s3_bucket, Key=file_key)
        except Exception as e:
            print(f"Error deleting file from S3: {e}")
    
    # Usuwanie publikacji
    await publications_collection.delete_one({"id": publication_id})
    return {"message": "Publication deleted successfully"}

@app.post("/publications/{publication_id}/file")
async def upload_publication_file(
    publication_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Pobieranie publikacji
    publication = await publications_collection.find_one({"id": publication_id})
    if publication is None:
        raise HTTPException(status_code=404, detail="Publication not found")
    
    # Sprawdzanie uprawnień
    if publication["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403