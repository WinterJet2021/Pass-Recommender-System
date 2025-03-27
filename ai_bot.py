from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import google.generativeai as genai
import os
import uuid

# ======================
# Configure Google Gemini API
# ======================
API_KEY = os.getenv("AIzaSyATtsrK61pPdCSpUeZRXlTee5TAePM6--M")
genai.configure(api_key=API_KEY)

router = APIRouter()

# Store user preferences (mock database for now)
user_sessions = {}

# ======================
# AI Chatbot WebSocket
# ======================
@router.websocket("/tuey/")
async def chatbot_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_id = str(uuid.uuid4())  # Generate unique user session ID
    user_sessions[user_id] = {"preferences": []}  # Initialize session

    try:
        await websocket.send_text("Hello! I can help you find events. What are your interests?")
        while True:
            user_message = await websocket.receive_text()
            response = process_user_message(user_id, user_message)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        if user_id in user_sessions:
            del user_sessions[user_id]  # Clean up session


def process_user_message(user_id, message):
    """Processes user input and refines recommendations dynamically."""
    user_sessions[user_id]["preferences"].append(message)
    prompt = f"""
    User preferences so far: {', '.join(user_sessions[user_id]['preferences'])}
    Suggest 3 events tailored to these interests.
    """

    try:
        response = genai.generate(prompt=prompt)
        if response and response.text:
            return response.text
        else:
            return "I couldn't find matching events. Try refining your preferences!"
    except Exception as e:
        return f"Error: {e}"