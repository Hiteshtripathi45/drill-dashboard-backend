from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow React (localhost:3000 or Vercel) to call API

# LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load models
model_path = os.path.join(os.path.dirname(__file__), 'rul_model.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
rul_model = LSTMModel(input_size=7, hidden_size=50, num_layers=2)
rul_model.load_state_dict(torch.load(model_path))
rul_model.eval()

temp_model_path = os.path.join(os.path.dirname(__file__), 'temp_est_model.joblib')
temp_scaler_path = os.path.join(os.path.dirname(__file__), 'temp_scaler.joblib')
if not os.path.exists(temp_model_path) or not os.path.exists(temp_scaler_path):
    raise FileNotFoundError("Joblib model or scaler not found")
temp_est_model = joblib.load(temp_model_path)
temp_scaler = joblib.load(temp_scaler_path)

@app.route('/')
def home():
    return jsonify({'status': 'Flask server running', 'endpoints': ['/predict_rul', '/predict_temp']})

@app.route('/predict_rul', methods=['POST'])
def predict_rul():
    if not request.is_json or 'sequence' not in request.json:
        return jsonify({'error': 'Invalid input: sequence required'}), 400
    data = request.json['sequence']  # [batch=1, seq=20, features=7]
    try:
        input_tensor = torch.tensor(np.array(data), dtype=torch.float32)
        with torch.no_grad():
            rul = rul_model(input_tensor).item()
        return jsonify({'rul': rul})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_temp', methods=['POST'])
def predict_temp():
    if not request.is_json or 'sequence' not in request.json:
        return jsonify({'error': 'Invalid input: sequence required'}), 400
    data = request.json['sequence']  # Flattened [120 features]
    try:
        input_flat = np.array(data).reshape(1, -1)
        pred_scaled = temp_est_model.predict(input_flat)[0]
        temp_est = temp_scaler.inverse_transform([[pred_scaled]])[0][0]  # Unscale
        return jsonify({'temp_est': temp_est})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)