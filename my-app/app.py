from model import preprocess_text,tfidf_vectorizer,nltk_classifier
import os
from flask import Flask, request,jsonify
from dotenv import load_dotenv

app = Flask("NLP")

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
port = os.getenv('PORT', None)

#Get endpoint
@app.route('/', methods = ['GET'])
def hello():
      return "Hello world",200

#Post endpoint
@app.route('/', methods=['POST'])
def commandHandler():
    data = request.get_json()
    print(">>>ReceivedData: ",data["sms"])

    preprocessed_text = preprocess_text(data["sms"])

    # Vectorize the preprocessed text using the same TF-IDF vectorizer
    new_text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

    # Convert the sparse matrix to a dense array and reshape it
    new_text_tfidf = new_text_tfidf.toarray().reshape(1, -1)

    # Use the classifier to make predictions
    predicted_label = nltk_classifier.classify(new_text_tfidf[0])

    if predicted_label == 0 :
        res = "This is NOT spam SMS"
    else:
        res = "This is spam SMS!!!!!!"

    return jsonify(res), 200


if __name__ == "__main__":
    app.run(port=int(port), use_reloader = True, debug=True)