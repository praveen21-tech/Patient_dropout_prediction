# ======================================
# STEP 1: IMPORT LIBRARIES
# ======================================
import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ======================================
# STEP 2: CONNECT TO SQLITE DATABASE
# ======================================
db_path = "appointments.db"   # Ensure DB is in same folder
conn = sqlite3.connect(db_path)

print("Connected to database successfully")


# ======================================
# STEP 3: DATABASE VALIDATION
# ======================================

# Total number of records
total_rows = pd.read_sql(
    "SELECT COUNT(*) AS total_records FROM appointments",
    conn
)
print("\nTotal records in database:")
print(total_rows)

# Dropout distribution
dropout_distribution = pd.read_sql(
    """
    SELECT dropout, COUNT(*) AS count
    FROM appointments
    GROUP BY dropout
    """,
    conn
)
print("\nDropout distribution:")
print(dropout_distribution)

# Age range validation
age_stats = pd.read_sql(
    """
    SELECT MIN(age) AS min_age, MAX(age) AS max_age
    FROM appointments
    """,
    conn
)
print("\nAge range:")
print(age_stats)


# ======================================
# STEP 4: FETCH DATA FROM DATABASE
# ======================================
query = """
SELECT
    gender,
    age,
    scholarship,
    hipertension,
    diabetes,
    alcoholism,
    handcap,
    sms_received,
    waiting_days,
    dropout
FROM appointments
"""

df = pd.read_sql(query, conn)
conn.close()

print("\nData fetched from database")
print(df.head())


# ======================================
# STEP 5: DEFINE FEATURES & TARGET
# ======================================
X = df.drop("dropout", axis=1)
y = df["dropout"]


# ======================================
# STEP 6: TRAIN-TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed")


# ======================================
# STEP 7: TRAIN RANDOM FOREST MODEL
# ======================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully")


# ======================================
# STEP 8: MODEL EVALUATION
# ======================================
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ======================================
# STEP 9: FEATURE IMPORTANCE
# ======================================
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)


# ======================================
# STEP 10: PREDICT DROPOUT PROBABILITY
# ======================================
df["dropout_probability"] = rf_model.predict_proba(X)[:, 1]

print("\nSample predictions with probabilities:")
print(df.head())
