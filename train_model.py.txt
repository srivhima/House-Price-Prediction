import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("housing.csv")

# Features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")