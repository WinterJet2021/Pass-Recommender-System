# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from algorithm import refine_with_gemini, event_df  # Import functions from algorithm.py

app = FastAPI()

# ==============================
#  Enable CORS Middleware
# ==============================
origins = ["https://digitalnomadsync.com", "http://digitalnomadsync.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
#  Request Body Model
# ==============================
class UserRequest(BaseModel):
    user_interests: List[str]
    location: str

# ==============================
#  Define API Endpoint
# ==============================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Digital Nomad Sync API!"}

@app.post("/pass/")
def get_recommendations(user_request: UserRequest):
    recommendations = refine_with_gemini(
        user_interests=user_request.user_interests,
        location=user_request.location,
        event_df=event_df
    )

    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found.")

    return {"Enhanced Recommendations": recommendations}