from fastapi import HTTPException
from pydantic import BaseModel  # Include this import
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import ast

# Define a new request model specific to time_recommendation.py
class TimeUserRequest(BaseModel):
    user_id: int

def prepare_user_data_for_matching(df):
    activity_cols = ['Basketball', 'Yoga', 'Hiking', 'Cycling', 'Gym', 
                     'Swimming', 'Dancing', 'Running', 'Music', 'Photography']
    activity_matrix = df[activity_cols].values.astype(float)

    df['Availability_Vector'] = df['Availability_Vector'].apply(ast.literal_eval)
    availability_matrix = np.array(df['Availability_Vector'].tolist()).astype(float)

    all_languages = sorted({lang for sublist in df['Language_IDs'] for lang in sublist})
    language_matrix = np.zeros((len(df), len(all_languages)))
    for i, langs in enumerate(df['Language_IDs']):
        for lang in langs:
            language_matrix[i, all_languages.index(lang)] = 1
    language_matrix = language_matrix.astype(float)

    location_encoder = OneHotEncoder(sparse_output=False)
    location_matrix = location_encoder.fit_transform(df[['Location_ID']]).astype(float)

    user_score_scaled = MinMaxScaler().fit_transform(df[['User Score']]).astype(float)

    nationality_encoder = OneHotEncoder(sparse_output=False)
    nationality_matrix = nationality_encoder.fit_transform(df[['Nationality_ID']]).astype(float)

    gender_matrix = df[['Gender']].values.astype(float)

    combined_features = np.hstack([
        activity_matrix * 3.0,
        availability_matrix * 2.0,
        language_matrix * 2.0,
        location_matrix * 1.5,
        user_score_scaled * 1.5,
        nationality_matrix * 0.5,
        gender_matrix * 0.5
    ])

    return combined_features, df['User_ID']


def match_users_by_time(user_id: int):
    try:
        df = pd.read_csv("preprocessed_50user_data.csv")

        user_vectors, user_ids = prepare_user_data_for_matching(df)
        similarity_matrix = cosine_similarity(user_vectors)

        matches = {}
        for idx, user_id_in_data in enumerate(user_ids):
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            matched_users = [(int(user_ids[i]), score) for i, score in sim_scores[1:6]]
            matches[int(user_id_in_data)] = matched_users

        if user_id not in matches:
            raise HTTPException(status_code=404, detail="User not found")

        recommendations = matches[user_id]
        results = [{"Matched_User_ID": match[0], "Match_Score": match[1]} for match in recommendations]

        return {"User_ID": user_id, "Matched_Users": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))