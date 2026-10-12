import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import math

# Step 1: Load dataset
df = pd.read_csv("flight_data.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Show column names
print("Columns available:", df.columns.tolist())
print(df.head())

# Helper function to get correct column name
cols = {c.lower(): c for c in df.columns}
def get_col(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    raise KeyError(f"None of {names} found in dataset columns {list(df.columns)}")

# Automatically detect correct columns
col_airline = get_col('Airline', 'Airline Name', 'Airline_Name')
col_price = get_col('Price', 'Ticket Price')
col_days_left = get_col('Days Left', 'Days_Left', 'days_left')
col_class = get_col('Class', 'Seat Class', 'seat_class')
col_source = get_col('Source City', 'Source', 'Source_City')
col_dest = get_col('Destination City', 'Destination', 'Destination_City')

# Step 2: Quick exploration
print(df.info())
print(df.describe())

# Handle missing values
df.dropna(inplace=True)

# Step 3: Data Visualization
plt.figure(figsize=(12,6))
sns.boxplot(x=col_airline, y=col_price, data=df)
plt.xticks(rotation=90)
plt.title("Ticket Price Variation by Airline")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=col_days_left, y=col_price, data=df)
plt.title("Price vs Days Left")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=col_class, data=df)
plt.title("Class Distribution")
plt.show()

# Step 4: One-Hot Encoding for categorical features
categorical_features = [col_airline, col_source, col_dest, col_class]
X = df.drop(columns=[col_price])
y = df[col_price]

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), categorical_features)
], remainder='passthrough')

X_encoded = ct.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Step 6: Standardize
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate model
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, MAPE: {mape*100:.2f}%, R2: {r2:.2f}")

# Step 7: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model(lr, X_test, y_test, "Linear Regression")

# Step 8: Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
evaluate_model(dt, X_test, y_test, "Decision Tree")

# Step 9: Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
evaluate_model(rf, X_test, y_test, "Random Forest")
