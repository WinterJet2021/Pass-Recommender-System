# algorithm.py

import numpy as np
import pandas as pd
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
#  Configure Google Gemini API
# ==============================
API_KEY = "AIzaSyATtsrK61pPdCSpUeZRXlTee5TAePM6--M"
genai.configure(api_key=API_KEY)

# ==============================
#  Load and Preprocess Data
# ==============================
file_path = 'Final_Thailand_Event_Recommendation_Dataset__Unique_Events_Everywhere_.csv'
event_df = pd.read_csv(file_path)

# Drop unnecessary columns
ml_event_df = event_df.drop(columns=["Event Description", "Date & Time", "Country"])

# Encode categorical columns
categorical_columns = ["Event Name", "Event Type", "City", "Cost", "Target Audience"]
label_encoders = {col: LabelEncoder().fit(ml_event_df[col]) for col in categorical_columns}
for col in categorical_columns:
    ml_event_df[col] = label_encoders[col].transform(ml_event_df[col])

# Normalize numerical columns
scaler = MinMaxScaler()
numerical_columns = ["Duration (hrs)", "Latitude", "Longitude", "Attendees", "Average Rating", "Review Count"]
ml_event_df[numerical_columns] = scaler.fit_transform(ml_event_df[numerical_columns])

# ==============================
#  Generate User Interactions
# ==============================
num_users = 100
num_events = len(ml_event_df)

actual_interactions = np.zeros((num_users, num_events))

event_popularity = ml_event_df.groupby("Event Type")["Attendees"].sum() + \
                   (ml_event_df.groupby("Event Type")["Average Rating"].mean() * 10)
event_popularity = event_popularity / event_popularity.sum()
event_types = ml_event_df["Event Type"].unique()
event_popularity = event_popularity.reindex(event_types, fill_value=0)
user_preferences = np.random.choice(event_types, size=num_users, p=event_popularity / event_popularity.sum())

for user in range(num_users):
    preferred_events = ml_event_df[ml_event_df["Event Type"] == user_preferences[user]].index.tolist()
    if preferred_events:
        num_attended = max(1, int(len(preferred_events) * np.random.uniform(0.6, 0.85)))
        attended_events = np.random.choice(preferred_events, size=num_attended, replace=False)
        actual_interactions[user, attended_events] = 1

# ==============================
#  Compute Similarity Matrices
# ==============================
event_similarity = cosine_similarity(ml_event_df.drop(columns=["Event Name", "City"]))
interaction_matrix = csr_matrix(actual_interactions)
event_interaction_similarity = cosine_similarity(interaction_matrix.T)

alpha = 0.4
hybrid_scores = (
    alpha * np.power(event_similarity, 1.2) +
    (1 - alpha) * np.power(event_interaction_similarity, 1.5)
)

# ==============================
#  Google Gemini NLP Enhancement
# ==============================
def refine_with_gemini(user_interests, location, event_df):
    filtered_events_df = event_df[event_df['City'].str.lower() == location.lower()]

    if filtered_events_df.empty:
        return ["No suitable events found in the specified location."]

    event_list_str = "\n".join(filtered_events_df['Event Name'].tolist())
    prompt = f"""
    Based on the user's interests: {', '.join(user_interests)} and location: {location},
    recommend 5 events from the list below with explanations of why they are relevant:
    {event_list_str}
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        if response and response.text:
            recommendations = response.text.split("\n")
            return [event.strip() for event in recommendations if event.strip()]
    except Exception as e:
        return [f"Error: {e}"]
