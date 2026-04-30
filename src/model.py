import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict_sentiment(model, text):
    return model.predict([text])[0]