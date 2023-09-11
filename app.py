from model import preprocess_text,tfidf_vectorizer,nltk_classifier

# Preprocess the new text data (replace 'new_text' with your actual text)
while 1:
    new_text = input("Paste your SMS here: ")
    preprocessed_text = preprocess_text(new_text)

    # Vectorize the preprocessed text using the same TF-IDF vectorizer
    new_text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

    # Convert the sparse matrix to a dense array and reshape it
    new_text_tfidf = new_text_tfidf.toarray().reshape(1, -1)

    # Use the classifier to make predictions
    predicted_label = nltk_classifier.classify(new_text_tfidf[0])

    # Print the predicted label
    if predicted_label == 0 :
        print("This is NOT spam SMS")
    else:
        print("This is spam SMS!!!!!!")
