import pickle
from preprocessing import clean_text

model = pickle.load(open("../models/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

def predict_spam(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    msg = input("Enter message: ")
    print("Prediction:", predict_spam(msg))