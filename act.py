import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
data = pd.read_csv('terrorism_data_large.csv')

# Clean missing values in relevant columns
data = data.dropna(subset=['region', 'weapon_type', 'attack_type'])

print("\n--- Preparing the Model ---")
choice = input("Enter your choice (1. Predict Attack Type, 2. Predict Weapon Type, 3. Predict Region): ")

if choice == '1':
    X = data[['region', 'weapon_type']]
    y = data['attack_type']
    input_cols = ['region', 'weapon_type']
elif choice == '2':
    X = data[['region', 'attack_type']]
    y = data['weapon_type']
    input_cols = ['region', 'attack_type']
elif choice == '3':
    X = data[['attack_type', 'weapon_type']]
    y = data['region']
    input_cols = ['attack_type', 'weapon_type']
else:
    print("Invalid choice.")
    exit()

target = y.name  # Save target name for later use

# Define preprocessing and model pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), input_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Data Overview
print("\n--- Data Overview ---")
print("First 5 Rows:\n", data.head())
print("\nData Summary:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# Plot: Distribution of Attack Types
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='attack_type', palette='Set2')
plt.title('Distribution of Attack Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: Number of Attacks by Region
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='region', palette='viridis')
plt.title('Number of Attacks by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap: Attack Type vs Weapon Type
heatmap_data = pd.crosstab(data['attack_type'], data['weapon_type'])
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Attack Type vs Weapon Type')
plt.tight_layout()
plt.show()

# Make a Prediction based on user input
print("\n--- Make a Prediction ---")
user_input = {col: [input(f"Enter {col}: ")] for col in input_cols}
input_df = pd.DataFrame(user_input)
prediction = model.predict(input_df)
print(f"\nPredicted {target}: {prediction[0]}")
