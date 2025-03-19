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
        async with httpx.AsyncClient() as client:
            # Call auth service to validate token
            response = await client.get(
                f"{os.environ.get('AUTH_SERVICE_URL', 'http://auth-service:8000')}/validate-token",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Content moderation function
async def moderate_content(text: str) -> bool:
    # Preprocess text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        is_inappropriate = predictions[:, 1].item() > 0.7  # Class 1 = inappropriate
    
    return is_inappropriate

# AI reply generation function
async def generate_ai_reply(post_content: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{os.environ.get('MODEL_HOSTING_URL', 'http://model-hosting-service:8000')}/generate",
                json={"prompt": post_content, "max_tokens": 500}
            )
            if response.status_code == 200:
                return response.json().get("generated_text", "")
            else:
                return "Sorry, I couldn't generate a response at this time."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# FORUM ENDPOINTS

# Create post
@app.post("/posts", response_model=Post)
async def create_post(post: PostCreate, current_user: Dict = Depends(get_current_user)):
    # Create post object
    post_dict = Post(
        title=post.title,
        content=post.content,
        tags=post.tags,
        user_id=current_user["id"],
        user_name=current_user["username"]
    ).dict()
    
    # Check for inappropriate content
    is_flagged = await moderate_content(post.content)
    post_dict["is_flagged"] = is_flagged
    post_dict["is_moderated"] = True
    
    # Generate AI reply if content is appropriate
    if not is_flagged:
        ai_reply = await generate_ai_reply(post.content)
        post_dict["ai_generated_reply"] = ai_reply
    
    # Insert into database
    result = await posts_collection.insert_one(post_dict)
    post_dict["id"] = str(result.inserted_id)
    
    # Notify recommendation service
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/events",
                json={"event_type": "post_created", "user_id": current_user["id"], "post_id": post_dict["id"]}
            )
    except Exception:
        # Log error but continue processing
        pass
    
    return post_dict

# Get post by ID
@app.get("/posts/{post_id}", response_model=Post)
async def get_post(post_id: str, current_user: Dict = Depends(get_current_user)):
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment view count
    await posts_collection.update_one(
        {"id": post_id},
        {"$inc": {"views": 1}}
    )
    post["views"] += 1
    
    # Record view for recommendation engine
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/events",
                json={"event_type": "post_viewed", "user_id": current_user["id"], "post_id": post_id}
            )
    except Exception:
        # Log error but continue processing
        pass
    
    return post

# Update post
@app.put("/posts/{post_id}", response_model=Post)
async def update_post(post_id: str, post_update: PostUpdate, current_user: Dict = Depends(get_current_user)):
    # Get existing post
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check ownership
    if post["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to update this post")
    
    # Update fields
    update_data = {k: v for k, v in post_update.dict().items() if v is not None}
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        
        # Check content for moderation if it was updated
        if "content" in update_data:
            is_flagged = await moderate_content(update_data["content"])
            update_data["is_flagged"] = is_flagged
            update_data["is_moderated"] = True
        
        await posts_collection.update_one(
            {"id": post_id},
            {"$set": update_data}
        )
    
    # Get updated post
    updated_post = await posts_collection.find_one({"id": post_id})
    return updated_post

# Delete post
@app.delete("/posts/{post_id}", status_code=204)
async def delete_post(post_id: str, current_user: Dict = Depends(get_current_user)):
    # Get post
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check ownership or admin
    if post["user_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this post")
    
    # Delete post and related comments
    await posts_collection.delete_one({"id": post_id})
    await comments_collection.delete_many({"post_id": post_id})
    
    # Notify recommendation service
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/events",
                json={"event_type": "post_deleted", "post_id": post_id}
            )
    except Exception:
        # Log error but continue processing
        pass

# List posts with pagination and filtering
@app.get("/posts", response_model=List[Post])
async def list_posts(
    skip: int = 0,
    limit: int = 20,
    tag: Optional[str] = None,
    user_id: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: int = -1,  # -1 for descending, 1 for ascending
    current_user: Dict = Depends(get_current_user)
):
    # Build query
    query = {}
    if tag:
        query["tags"] = tag
    if user_id:
        query["user_id"] = user_id
    
    # Hide flagged content for regular users
    if current_user.get("role") != "admin":
        query["is_flagged"] = {"$ne": True}
    
    # Fetch posts
    cursor = posts_collection.find(query)
    cursor = cursor.sort(sort_by, sort_order).skip(skip).limit(limit)
    posts = await cursor.to_list(length=limit)
    
    return posts

# Add comment
@app.post("/posts/{post_id}/comments", response_model=Comment)
async def add_comment(
    post_id: str,
    comment: CommentCreate,
    current_user: Dict = Depends(get_current_user)
):
    # Check if post exists
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Create comment object
    comment_dict = Comment(
        post_id=post_id,
        content=comment.content,
        parent_id=comment.parent_id,
        user_id=current_user["id"],
        user_name=current_user["username"]
    ).dict()
    
    # Check for inappropriate content
    is_flagged = await moderate_content(comment.content)
    comment_dict["is_flagged"] = is_flagged
    comment_dict["is_moderated"] = True
    
    # Insert into database
    result = await comments_collection.insert_one(comment_dict)
    comment_dict["id"] = str(result.inserted_id)
    
    # Notify users
    try:
        # Notify post author
        if post["user_id"] != current_user["id"]:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{os.environ.get('NOTIFICATION_SERVICE_URL', 'http://notification-service:8000')}/notifications",
                    json={
                        "user_id": post["user_id"],
                        "type": "comment",
                        "content": f"{current_user['username']} commented on your post: {post['title']}",
                        "link": f"/posts/{post_id}"
                    }
                )
        
        # Notify parent comment author if applicable
        if comment.parent_id:
            parent_comment = await comments_collection.find_one({"id": comment.parent_id})
            if parent_comment and parent_comment["user_id"] != current_user["id"]:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{os.environ.get('NOTIFICATION_SERVICE_URL', 'http://notification-service:8000')}/notifications",
                        json={
                            "user_id": parent_comment["user_id"],
                            "type": "reply",
                            "content": f"{current_user['username']} replied to your comment",
                            "link": f"/posts/{post_id}"
                        }
                    )
    except Exception:
        # Log error but continue processing
        pass
    
    return comment_dict

# Get comments for a post
@app.get("/posts/{post_id}/comments", response_model=List[Comment])
async def get_comments(post_id: str, current_user: Dict = Depends(get_current_user)):
    # Build query
    query = {"post_id": post_id}
    
    # Hide flagged content for regular users
    if current_user.get("role") != "admin":
        query["is_flagged"] = {"$ne": True}
    
    # Fetch comments
    cursor = comments_collection.find(query).sort("created_at", 1)
    comments = await cursor.to_list(length=1000)  # Reasonable limit
    
    return comments

# Vote on post
@app.post("/posts/{post_id}/vote")
async def vote_post(
    post_id: str,
    vote_type: str,  # "up" or "down"
    current_user: Dict = Depends(get_current_user)
):
    # Validate vote type
    if vote_type not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="Invalid vote type")
    
    # Check if post exists
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Update vote count
    field = "upvotes" if vote_type == "up" else "downvotes"
    await posts_collection.update_one(
        {"id": post_id},
        {"$inc": {field: 1}}
    )
    
    # Record vote for recommendation engine
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/events",
                json={
                    "event_type": f"post_{vote_type}voted",
                    "user_id": current_user["id"],
                    "post_id": post_id
                }
            )
    except Exception:
        # Log error but continue processing
        pass
    
    return {"success": True}

# CHAT ENDPOINTS

# Create chat room
@app.post("/chat/rooms", response_model=ChatRoom)
async def create_chat_room(
    room: ChatRoomCreate,
    current_user: Dict = Depends(get_current_user)
):
    # Create room with creator as first member
    room_dict = ChatRoom(
        name=room.name,
        description=room.description,
        user_ids=[current_user["id"]] + [uid for uid in room.user_ids if uid != current_user["id"]],
        is_private=room.is_private
    ).dict()
    
    # Insert into database
    result = await chat_rooms_collection.insert_one(room_dict)
    room_dict["id"] = str(result.inserted_id)
    
    return room_dict

# Get chat room
@app.get("/chat/rooms/{room_id}", response_model=ChatRoom)
async def get_chat_room(room_id: str, current_user: Dict = Depends(get_current_user)):
    room = await chat_rooms_collection.find_one({"id": room_id})
    if room is None:
        raise HTTPException(status_code=404, detail="Chat room not found")
    
    # Check access for private rooms
    if room["is_private"] and current_user["id"] not in room["user_ids"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat room")
    
    return room

# List chat rooms
@app.get("/chat/rooms", response_model=List[ChatRoom])
async def list_chat_rooms(current_user: Dict = Depends(get_current_user)):
    # Build query - show public rooms and private rooms the user is in
    query = {
        "$or": [
            {"is_private": False},
            {"is_private": True, "user_ids": current_user["id"]}
        ]
    }
    
    # Fetch rooms
    cursor = chat_rooms_collection.find(query).sort("updated_at", -1)
    rooms = await cursor.to_list(length=100)
    
    return rooms

# Join chat room
@app.post("/chat/rooms/{room_id}/join")
async def join_chat_room(room_id: str, current_user: Dict = Depends(get_current_user)):
    room = await chat_rooms_collection.find_one({"id": room_id})
    if room is None:
        raise HTTPException(status_code=404, detail="Chat room not found")
    
    # Check if private
    if room["is_private"]:
        raise HTTPException(status_code=403, detail="Cannot join private chat room")
    
    # Add user to room if not already a member
    if current_user["id"] not in room["user_ids"]:
        await chat_rooms_collection.update_one(
            {"id": room_id},
            {
                "$push": {"user_ids": current_user["id"]},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
    
    return {"success": True}

# Get chat messages
@app.get("/chat/rooms/{room_id}/messages", response_model=List[ChatMessage])
async def get_chat_messages(
    room_id: str,
    limit: int = 50,
    before: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    # Check room exists and user has access
    room = await chat_rooms_collection.find_one({"id": room_id})
    if room is None:
        raise HTTPException(status_code=404, detail="Chat room not found")
    
    if room["is_private"] and current_user["id"] not in room["user_ids"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat room")
    
    # Build query
    query = {"room_id": room_id}
    
    # Add pagination
    if before:
        before_msg = await chat_messages_collection.find_one({"id": before})
        if before_msg:
            query["created_at"] = {"$lt": before_msg["created_at"]}
    
    # Hide flagged content for regular users
    if current_user.get("role") != "admin":
        query["is_flagged"] = {"$ne": True}
    
    # Fetch messages
    cursor = chat_messages_collection.find(query).sort("created_at", -1).limit(limit)
    messages = await cursor.to_list(length=limit)
    
    # Return messages in chronological order
    messages.reverse()
    
    return messages

# WebSocket endpoint for chat
@app.websocket("/chat/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, token: str):
    # Validate user token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('AUTH_SERVICE_URL', 'http://auth-service:8000')}/validate-token",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code != 200:
                await websocket.close(code=1008)
                return
            user = response.json()
    except Exception:
        await websocket.close(code=1008)
        return
    
    # Check room exists and user has access
    room = await chat_rooms_collection.find_one({"id": room_id})
    if room is None:
        await websocket.close(code=1008)
        return
    
    if room["is_private"] and user["id"] not in room["user_ids"]:
        await websocket.close(code=1008)
        return
    
    # Accept connection
    await manager.connect(websocket, room_id)
    
    try:
        # Notify room about new connection
        connection_message = {
            "type": "system",
            "content": f"{user['username']} joined the chat",
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast(json.dumps(connection_message), room_id)
        
        # Process messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Validate message
            if "content" not in message_data or not message_data["content"].strip():
                continue
            
            # Check content for moderation
            is_flagged = await moderate_content(message_data["content"])
            
            # Store message
            chat_message = ChatMessage(
                room_id=room_id,
                content=message_data["content"],
                user_id=user["id"],
                user_name=user["username"],
                is_flagged=is_flagged,
                is_moderated=True
            ).dict()
            
            await chat_messages_collection.insert_one(chat_message)
            
            # Broadcast message if not flagged
            if not is_flagged:
                broadcast_data = {
                    "type": "message",
                    "id": chat_message["id"],
                    "content": chat_message["content"],
                    "user_id": chat_message["user_id"],
                    "user_name": chat_message["user_name"],
                    "timestamp": chat_message["created_at"].isoformat()
                }
                await manager.broadcast(json.dumps(broadcast_data), room_id)
            else:
                # Send private message to user about flagged content
                flagged_notification = {
                    "type": "system",
                    "content": "Your message was flagged for inappropriate content and won't be shown to other users.",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(flagged_notification))
                
                # Notify admins if any are in the room
                async for admin_user in chat_rooms_collection.aggregate([
                    {"$match": {"id": room_id}},
                    {"$lookup": {
                        "from": "users",
                        "localField": "user_ids",
                        "foreignField": "id",
                        "as": "users"
                    }},
                    {"$unwind": "$users"},
                    {"$match": {"users.role": "admin"}},
                    {"$project": {"users.id": 1}}
                ]):
                    try:
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                f"{os.environ.get('NOTIFICATION_SERVICE_URL', 'http://notification-service:8000')}/notifications",
                                json={
                                    "user_id": admin_user["users"]["id"],
                                    "type": "moderation",
                                    "content": f"Flagged message in room {room['name']} by {user['username']}",
                                    "link": f"/chat/rooms/{room_id}"
                                }
                            )
                    except Exception:
                        pass
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        
        # Notify room about disconnection
        disconnection_message = {
            "type": "system",
            "content": f"{user['username']} left the chat",
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast(json.dumps(disconnection_message), room_id)
    except Exception as e:
        manager.disconnect(websocket, room_id)
        print(f"WebSocket error: {str(e)}")

# ADMIN ENDPOINTS

# Moderate post
@app.put("/admin/posts/{post_id}/moderate", response_model=Post)
async def moderate_post(
    post_id: str,
    is_flagged: bool,
    current_user: Dict = Depends(get_current_user)
):
    # Check admin
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update post
    await posts_collection.update_one(
        {"id": post_id},
        {
            "$set": {
                "is_flagged": is_flagged,
                "is_moderated": True,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Get updated post
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    return post

# Moderate comment
@app.put("/admin/comments/{comment_id}/moderate", response_model=Comment)
async def moderate_comment(
    comment_id: str,
    is_flagged: bool,
    current_user: Dict = Depends(get_current_user)
):
    # Check admin
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update comment
    await comments_collection.update_one(
        {"id": comment_id},
        {
            "$set": {
                "is_flagged": is_flagged,
                "is_moderated": True,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Get updated comment
    comment = await comments_collection.find_one({"id": comment_id})
    if comment is None:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    return comment

# Get flagged content
@app.get("/admin/moderation/flagged", response_model=Dict)
async def get_flagged_content(current_user: Dict = Depends(get_current_user)):
    # Check admin
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get flagged posts
    flagged_posts_cursor = posts_collection.find({"is_flagged": True}).sort("created_at", -1).limit(50)
    flagged_posts = await flagged_posts_cursor.to_list(length=50)
    
    # Get flagged comments
    flagged_comments_cursor = comments_collection.find({"is_flagged": True}).sort("created_at", -1).limit(50)
    flagged_comments = await flagged_comments_cursor.to_list(length=50)
    
    # Get flagged chat messages
    flagged_messages_cursor = chat_messages_collection.find({"is_flagged": True}).sort("created_at", -1).limit(50)
    flagged_messages = await flagged_messages_cursor.to_list(length=50)
    
    return {
        "posts": flagged_posts,
        "comments": flagged_comments,
        "chat_messages": flagged_messages
    }

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "forum-chat"}

# Add startup event to create indexes
@app.on_event("startup")
async def create_indexes():
    # Create indexes for posts collection
    await posts_collection.create_index("id", unique=True)
    await posts_collection.create_index("user_id")
    await posts_collection.create_index("tags")
    await posts_collection.create_index("is_flagged")
    await posts_collection.create_index("created_at")
    
    # Create indexes for comments collection
    await comments_collection.create_index("id", unique=True)
    await comments_collection.create_index("post_id")
    await comments_collection.create_index("user_id")
    await comments_collection.create_index("parent_id")
    await comments_collection.create_index("is_flagged")
    
    # Create indexes for chat rooms collection
    await chat_rooms_collection.create_index("id", unique=True)
    await chat_rooms_collection.create_index("user_ids")
    await chat_rooms_collection.create_index("is_private")
    
    # Create indexes for chat messages collection
    await chat_messages_collection.create_index("id", unique=True)
    await chat_messages_collection.create_index("room_id")
    await chat_messages_collection.create_index("user_id")
    await chat_messages_collection.create_index("is_flagged")
    await chat_messages_collection.create_index("created_at")
    
    print("Database indexes created successfully")

# TRENDING AND RECOMMENDATION ENDPOINTS

# Get trending posts
@app.get("/trending/posts", response_model=List[Post])
async def get_trending_posts(
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    # Get trending posts from recommendation service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/trending/posts",
                params={"limit": limit, "user_id": current_user["id"]}
            )
            if response.status_code == 200:
                trending_data = response.json()
                post_ids = [item["post_id"] for item in trending_data]
                
                # Fetch posts
                posts = []
                for post_id in post_ids:
                    post = await posts_collection.find_one({"id": post_id, "is_flagged": {"$ne": True}})
                    if post:
                        posts.append(post)
                        
                return posts
    except Exception:
        pass
    
    # Fallback: return recent popular posts
    cursor = posts_collection.find({"is_flagged": {"$ne": True}})
    cursor = cursor.sort([("views", -1), ("created_at", -1)]).limit(limit)
    posts = await cursor.to_list(length=limit)
    
    return posts

# Get recommended posts for user
@app.get("/recommendations/posts", response_model=List[Post])
async def get_recommended_posts(
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    # Get recommended posts from recommendation service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/recommendations/posts",
                params={"limit": limit, "user_id": current_user["id"]}
            )
            if response.status_code == 200:
                rec_data = response.json()
                post_ids = [item["post_id"] for item in rec_data]
                
                # Fetch posts
                posts = []
                for post_id in post_ids:
                    post = await posts_collection.find_one({"id": post_id, "is_flagged": {"$ne": True}})
                    if post:
                        posts.append(post)
                        
                return posts
    except Exception:
        pass
    
    # Fallback: return recent posts
    cursor = posts_collection.find({"is_flagged": {"$ne": True}})
    cursor = cursor.sort("created_at", -1).limit(limit)
    posts = await cursor.to_list(length=limit)
    
    return posts

# Get similar posts (for post detail page)
@app.get("/posts/{post_id}/similar", response_model=List[Post])
async def get_similar_posts(
    post_id: str,
    limit: int = 5,
    current_user: Dict = Depends(get_current_user)
):
    # Check if post exists
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Get similar posts from recommendation service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/similar/posts",
                params={"post_id": post_id, "limit": limit}
            )
            if response.status_code == 200:
                similar_data = response.json()
                similar_post_ids = [item["post_id"] for item in similar_data]
                
                # Fetch posts
                similar_posts = []
                for similar_id in similar_post_ids:
                    if similar_id != post_id:  # Don't include the original post
                        similar_post = await posts_collection.find_one({"id": similar_id, "is_flagged": {"$ne": True}})
                        if similar_post:
                            similar_posts.append(similar_post)
                            
                return similar_posts[:limit]
    except Exception:
        pass
    
    # Fallback: return posts with similar tags
    if post.get("tags"):
        cursor = posts_collection.find({
            "id": {"$ne": post_id},  # Exclude current post
            "tags": {"$in": post["tags"]},
            "is_flagged": {"$ne": True}
        })
        cursor = cursor.sort("views", -1).limit(limit)
        similar_posts = await cursor.to_list(length=limit)
        return similar_posts
    
    # Ultimate fallback: return recent popular posts
    cursor = posts_collection.find({
        "id": {"$ne": post_id},  # Exclude current post
        "is_flagged": {"$ne": True}
    })
    cursor = cursor.sort([("views", -1), ("created_at", -1)]).limit(limit)
    similar_posts = await cursor.to_list(length=limit)
    
    return similar_posts

# SEARCH FUNCTIONALITY

# Search posts
@app.get("/search/posts", response_model=List[Post])
async def search_posts(
    query: str,
    tags: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: Dict = Depends(get_current_user)
):
    # Parse tags if provided
    tag_list = tags.split(",") if tags else []
    
    # Build query
    search_query = {
        "$and": [
            {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"content": {"$regex": query, "$options": "i"}}
                ]
            },
            {"is_flagged": {"$ne": True}}
        ]
    }
    
    # Add tags filter if provided
    if tag_list:
        search_query["$and"].append({"tags": {"$all": tag_list}})
    
    # Fetch posts
    cursor = posts_collection.find(search_query)
    cursor = cursor.sort("created_at", -1).skip(skip).limit(limit)
    posts = await cursor.to_list(length=limit)
    
    # Send search query to recommendation service for analytics
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{os.environ.get('RECOMMENDATION_SERVICE_URL', 'http://recommendation-service:8000')}/events",
                json={
                    "event_type": "search_performed",
                    "user_id": current_user["id"],
                    "query": query,
                    "tags": tag_list
                }
            )
    except Exception:
        # Log error but continue processing
        pass
    
    return posts

# REPORTS AND ANALYTICS

# Get post analytics
@app.get("/posts/{post_id}/analytics")
async def get_post_analytics(post_id: str, current_user: Dict = Depends(get_current_user)):
    # Check if post exists
    post = await posts_collection.find_one({"id": post_id})
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check ownership or admin
    if post["user_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view post analytics")
    
    # Get analytics from monetization service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('MONETIZATION_SERVICE_URL', 'http://monetization-service:8000')}/analytics/post/{post_id}"
            )
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    
    # Basic analytics fallback
    comment_count = await comments_collection.count_documents({"post_id": post_id})
    
    return {
        "views": post.get("views", 0),
        "upvotes": post.get("upvotes", 0),
        "downvotes": post.get("downvotes", 0),
        "comments": comment_count
    }

# USER ACTIVITY ENDPOINTS

# Get user activity
@app.get("/users/{user_id}/activity")
async def get_user_activity(
    user_id: str,
    skip: int = 0,
    limit: int = 20,
    current_user: Dict = Depends(get_current_user)
):
    # Get user posts
    posts_cursor = posts_collection.find({"user_id": user_id, "is_flagged": {"$ne": True}})
    posts_cursor = posts_cursor.sort("created_at", -1).skip(skip).limit(limit)
    posts = await posts_cursor.to_list(length=limit)
    
    # Get user comments
    comments_cursor = comments_collection.find({"user_id": user_id, "is_flagged": {"$ne": True}})
    comments_cursor = comments_cursor.sort("created_at", -1).skip(skip).limit(limit)
    comments = await comments_cursor.to_list(length=limit)
    
    # For comments, include the post title
    for comment in comments:
        post = await posts_collection.find_one({"id": comment["post_id"]})
        if post:
            comment["post_title"] = post["title"]
    
    return {
        "posts": posts,
        "comments": comments
    }

# TAGS ENDPOINTS

# Get popular tags
@app.get("/tags/popular", response_model=List[Dict])
async def get_popular_tags(limit: int = 20):
    # Aggregate to find most used tags
    pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {"tag": "$_id", "count": 1, "_id": 0}}
    ]
    
    cursor = posts_collection.aggregate(pipeline)
    tags = await cursor.to_list(length=limit)
    
    return tags

# METRICS ENDPOINT (For monitoring)

# Get service metrics
@app.get("/metrics")
async def get_metrics(current_user: Dict = Depends(get_current_user)):
    # Check if admin
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get total counts
    post_count = await posts_collection.count_documents({})
    comment_count = await comments_collection.count_documents({})
    chat_room_count = await chat_rooms_collection.count_documents({})
    chat_message_count = await chat_messages_collection.count_documents({})
    
    # Get counts for last 24 hours
    one_day_ago = datetime.utcnow() - timedelta(days=1)
    new_posts = await posts_collection.count_documents({"created_at": {"$gte": one_day_ago}})
    new_comments = await comments_collection.count_documents({"created_at": {"$gte": one_day_ago}})
    new_chat_messages = await chat_messages_collection.count_documents({"created_at": {"$gte": one_day_ago}})
    
    # Get flagged content counts
    flagged_posts = await posts_collection.count_documents({"is_flagged": True})
    flagged_comments = await comments_collection.count_documents({"is_flagged": True})
    flagged_messages = await chat_messages_collection.count_documents({"is_flagged": True})
    
    return {
        "total_posts": post_count,
        "total_comments": comment_count,
        "total_chat_rooms": chat_room_count,
        "total_chat_messages": chat_message_count,
        "new_posts_24h": new_posts,
        "new_comments_24h": new_comments,
        "new_chat_messages_24h": new_chat_messages,
        "flagged_posts": flagged_posts,
        "flagged_comments": flagged_comments,
        "flagged_messages": flagged_messages
    }

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)