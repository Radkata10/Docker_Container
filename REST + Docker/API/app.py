from flask import Flask, jsonify, request
from utilities import predict_iris
from pandas.api.types import is_numeric_dtype
import pandas as pd

app = Flask(__name__)


@app.post('/predict')
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    df.head()
    
    if df.shape[1] != 4:
        return jsonify({'AttributeError': 'Not enought measurements sent'})
    
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            return jsonify({'AttributeError': 'Non-numeric values'})
        
    if (df.values < 0).any():
        return jsonify({'AttributeError': 'Invalid values'})

    predictions = predict_iris(df.iloc[0, 0], df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3])
    try:
        result = jsonify(predictions)
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)