from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# NO TRY BLOCK HERE - This prevents line 10/11 errors
model = pickle.load(open('model.pkl', 'rb'))
print("Success: ML Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['avg']), 
            float(data['sr']), 
            float(data['recent'])
        ]
        prediction = model.predict([features])[0]
        probs = model.predict_proba([features])[0]
        confidence = round(max(probs) * 100, 2)
        
        return jsonify({
            'form': str(prediction),
            'confidence': f"{confidence}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)