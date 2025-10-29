
# ---

### 💻 **carbon_emission_predictor.py**

# ```python
"""
Predicting Carbon Emissions Using Machine Learning
Author: [Your Name]
SDG 13: Climate Action
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

# Load dataset
df = pd.read_csv('carbon_emissions.csv')

# Drop nulls
df = df.dropna(subset=['GDP', 'Energy_Consumption', 'Population', 'CO2_Emissions'])

# Define features and target
X = df[['GDP', 'Energy_Consumption', 'Population']]
y = df['CO2_Emissions']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot visualization
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual CO₂ Emissions')
plt.ylabel('Predicted CO₂ Emissions')
plt.title('Actual vs Predicted CO₂ Emissions')
plt.show()
