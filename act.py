import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('terrorism_data_large.csv')

# Initial Data Exploration
print("First 5 Rows of the Dataset:\n", data.head())
print("\nData Summary:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# Plot 1: Attack type distribution
plt.figure(figsize=(10,6))
sns.countplot(data=data, x='attack_type', hue='attack_type', palette='Set2', legend=False)
plt.title('Distribution of Attack Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Region-wise attack count
plt.figure(figsize=(10,6))
sns.countplot(data=data, x='region', hue='region', palette='viridis', legend=False)
plt.title('Number of Attacks by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Heatmap - Attack type vs Weapon type
heatmap_data = pd.crosstab(data['attack_type'], data['weapon_type'])
plt.figure(figsize=(10,8))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Attack Type vs Weapon Type')
plt.tight_layout()
plt.show()

# Features and target
X = data[['region', 'weapon_type']]
y = data['attack_type']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['region', 'weapon_type'])
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix (text only)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

