from flask import Flask, render_template, request
import pandas as pd
import pickle
import re

import pickle

# load the model from disk
filename = 'Sentiment_analysis.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transformer.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['content']
        data = message
        data = re.sub(r'@[A-Za-z0-9]+', '', data)
        data = re.sub(r'#', '', data)
        data = re.sub(r'RT[\s]', '', data)
        data = re.sub(r'https?:\/\/\S+', '', data)
        data = re.sub(r'[^\w\s]', '', data)
        data = re.sub(r'\n', '', data)
        data = re.sub(r'_', '', data)
        data = re.sub(" \d+", "", data)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        data = emoji_pattern.sub(r'', data)
        data = data.lower()
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)