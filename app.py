from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the saved model
model_filename = 'model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame(data)
        
        label_encoder=LabelEncoder()
        input_data['Gender']=label_encoder.fit_transform(input_data['Gender'])
        input_data['Symptoms_Present']=label_encoder.fit_transform(input_data['Symptoms_Present'])
        
        # Make predictions using the loaded model
        predictions = model.predict(input_data)
        
        # Return the predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask app
    app.run(port=5000, debug=True)
