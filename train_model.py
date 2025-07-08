import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("data.csv")

# Features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Preprocessing for categorical data
categorical_features = ['Education']
numeric_features = ['Experience', 'Age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "salary_model.pkl")

print("✅ Model trained and saved as salary_model.pkl")
