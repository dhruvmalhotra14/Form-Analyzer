import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Load your actual dataset
try:
    # Ensure cricket_data.csv is in the same folder as this script
    df = pd.read_csv('cricket_data.csv', encoding='utf-8') 
    print("Success: CSV loaded successfully!")
except FileNotFoundError:
    print("Error: 'cricket_data.csv' not found in the backend folder!")
    exit()

# 2. Pick the columns you want to use for prediction
# These must match the headers in your CSV exactly
features = ['Batting_Average', 'Strike_Rate', 'Recent_Form_Score']
X = df[features]
y = df['Player_Form']

# 3. Train the AI
model = RandomForestClassifier()
model.fit(X, y)

# 4. Save the 'knowledge' as a pkl file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ 'model.pkl' has been created in your folder!")