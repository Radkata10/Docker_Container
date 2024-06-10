import pickle

with open('models/model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)
    
def predict_iris(a, b, c, d):
    prediction = int(loaded_model.predict([[a, b, c, d]])[0])
    response = {"class": prediction}
    return response

# print(loaded_model.predict([[100, 100, 100, 232]]))