# %%
import pandas as pd

# Placeholder data format
data = pd.DataFrame({
    "survey_response": [
        "I love painting, attending art exhibitions and yoga",
        "Looking for hackathons or AI workshops in Thailand",
        "I enjoy hiking, rock climbing, and beach volleyball",
        "Foodie here! I want to go on culinary adventures"
    ],
    "labels": [
        ["Arts", "Wellness"],
        ["Technology"],
        ["Sports", "Travel"],
        ["Food", "Travel"]
    ]
})

# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['labels'])

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(data['survey_response'])

# %%
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

model = OneVsRestClassifier(LogisticRegression())
model.fit(X, y)

# %%
from sklearn.metrics import classification_report

y_pred = model.predict(X)
print(classification_report(y, y_pred, target_names=mlb.classes_))

# %%
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ------------------------------
# STEP 1: Load and clean dataset
# ------------------------------

# Load dataset
df = pd.read_csv("survey_interest_dataset_realistic.csv")

# Safely convert labels from strings to actual Python lists
def safe_literal_eval(val):
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except:
        return []

df['labels'] = df['labels'].apply(safe_literal_eval)

# Drop rows with empty or invalid labels
df = df[df['labels'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
print(f"✅ Dataset loaded with {len(df)} usable rows.")

# ------------------------------------------
# STEP 2: Binarize labels for multilabel ML
# ------------------------------------------

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['labels'])

# ------------------------------------------
# STEP 3: Train-test split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df['survey_answer'], Y, test_size=0.2, random_state=42
)

# ------------------------------------------
# STEP 4: TF-IDF + ML classifier pipeline
# ------------------------------------------

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
])

# Train a separate classifier for each label
from sklearn.multiclass import OneVsRestClassifier
model = OneVsRestClassifier(pipeline)
model.fit(X_train, y_train)

# ------------------------------------------
# STEP 5: Evaluate
# ------------------------------------------

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=mlb.classes_)
print("🎯 Classification Report:")
print(report)

# ------------------------------------------
# STEP 6: Try a sample prediction
# ------------------------------------------

sample_text = ["I’m looking for people to hike with and do yoga on the beach."]  # you can change this
pred = model.predict(sample_text)
decoded = mlb.inverse_transform(pred)

print("\n🧠 Predicted Interests:", decoded[0] if decoded[0] else ["No confident prediction"])


