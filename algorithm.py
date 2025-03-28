import numpy as np
import pandas as pd
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time  # For measuring response times

# ==============================
#  Configure Google Gemini API (Once)
# ==============================
API_KEY = "AIzaSyATtsrK61pPdCSpUeZRXlTee5TAePM6--M"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
except Exception as e:
    print(f"Error while configuring Gemini API: {e}")

# ==============================
#  Load and Preprocess Data
# ==============================
file_path = 'Thailand_Final_Realistic_Sports_Names_Corrected.csv'
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
#  Google Gemini NLP Enhancement
# ==============================
def refine_with_gemini(user_interests, location, event_df, max_events=20):
    start_time = time.time()  # Track the start time for performance measurement

    filtered_events_df = event_df[event_df['City'].str.lower() == location.lower()]

    if filtered_events_df.empty:
        return ["No suitable events found in the specified location."]

    # Limit the number of events to process to reduce latency
    filtered_events_df = filtered_events_df.sample(min(len(filtered_events_df), max_events))

    event_list_str = "\n".join(filtered_events_df['Event Name'].tolist())
    prompt = f"""
    The user is interested in: {', '.join(user_interests)} and is located in: {location}.
    Based on the following event list, recommend 5 events with a brief description of what each event is about:
    {event_list_str}
    """

    try:
        response = model.generate_content(prompt)
        if response and response.text:
            recommendations = response.text.split("\n")
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"API Response Time: {elapsed_time:.2f} seconds")
            return [event.strip() for event in recommendations if event.strip()]
    except Exception as e:
        return [f"Error: {e}"]

# ==============================
#  Measure Performance (Optional)
# ==============================
def measure_performance():
    import time
    start_time = time.time()
    refine_with_gemini(['Sports', 'Food'], 'Bangkok', event_df)
    print(f"Total Processing Time: {time.time() - start_time:.2f} seconds")