# services/forum-chat/forum_chat.py
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import uuid
import os
import json
import asyncio
import httpx
import motor.motor_asyncio
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Models
class Post(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    user_id: str
    user_name: str
    tags: List[str] = []
    views: int = 0
    upvotes: int = 0
    downvotes: int = 0
    ai_generated_reply: Optional[str] = None
    is_moderated: bool = False
    is_flagged: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PostCreate(BaseModel):
    title: str
    content: str
    tags: List[str] = []

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None

class Comment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    post_id: str
    content: str
    user_id: str
    user_name: str
    parent_id: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    is_moderated: bool = False
    is_flagged: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CommentCreate(BaseModel):
    content: str
    parent_id: Optional[str] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    room_id: str
    content: str
    user_id: str
    user_name: str
    is_moderated: bool = False
    is_flagged: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatRoom(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    user_ids: List[str] = []
    is_private: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ChatRoomCreate(BaseModel):
    name: str
    description: Optional[str] = None
    user_ids: List[str] = []
    is_private: bool = False

# Initialize FastAPI
app = FastAPI(title="Forum/Chat Service API")

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

# MongoDB configuration
mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
db = mongodb_client.forum_chat
posts_collection = db.posts
comments_collection = db.comments
chat_messages_collection = db.chat_messages
chat_rooms_collection = db.chat_rooms

# Load BERT model for content moderation
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, message: str, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)

manager = ConnectionManager()

# Function to verify user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        async with httpx.AsyncClient