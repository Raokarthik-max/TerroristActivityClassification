import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset (replace this with actual GTD data for real use)
data = pd.DataFrame({
    'region': ['South Asia', 'Middle East', 'South Asia', 'Sub-Saharan Africa', 'South Asia'],
    'weapon_type': ['Explosives', 'Firearms', 'Explosives', 'Melee', 'Firearms'],
    'attack_type': ['Bombing', 'Armed Assault', 'Bombing', 'Assassination', 'Armed Assault']
})

# Encode categorical features
le_region = LabelEncoder()
le_weapon = LabelEncoder()
le_attack = LabelEncoder()

data['region'] = le_region.fit_transform(data['region'])
data['weapon_type'] = le_weapon.fit_transform(data['weapon_type'])
data['attack_type'] = le_attack.fit_transform(data['attack_type'])  # target

# Features and target
X = data[['region', 'weapon_type']]
y = data['attack_type']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Train Random Forest model (or use DecisionTreeClassifier())
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=np.unique(y_test),
    target_names=le_attack.inverse_transform(np.unique(y_test))
))
