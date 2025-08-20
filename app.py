from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

spam_indicators = {
    r'free\s+money': 3,
    r'win\s+cash': 3,
    r'click\s+here': 2,
    r'urgent': 2,
    r'act\s+now': 2,
    r'limited\s+time': 2,
    r'risk\s+free': 2,
    r'credit\s+card': 2,
    r'bitcoin': 3,
    r'exclusive\s+deal': 2,
    r'congratulations': 2,
    r'loan\s+approval': 3,
    r'earn\s+\$\$\$': 3,
    r'claim\s+now': 3,
    r'buy\s+now': 2,
    r'lowest\s+price': 2,
    r'unsubscribe': 1,
    r'cheap\s+meds': 3,
    r'work\s+from\s+home': 2,
    r'https?://': 2,
    r'\$\d+': 2,
    r'\b\d{10,}\b': 1
}

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def calculate_rule_score(text):
    score = 0
    for pattern, weight in spam_indicators.items():
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
    return score

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email = request.form['email'].strip()

        rule_score = calculate_rule_score(email)
        print(f"Rule score: {rule_score}")

        email_processed = preprocess_text(email)
        email_vec = vectorizer.transform([email_processed])
        ml_probs = model.predict_proba(email_vec)
        ml_spam_prob = ml_probs[0, 0]
        print(f"ML spam prob: {ml_spam_prob}")

        combined_score = rule_score * 0.4 + ml_spam_prob * 0.6
        print(f"Combined score: {combined_score}")

        threshold = 1.5

        if combined_score >= threshold:
            prediction = f'SPAM (Advanced Hybrid) - Score: {combined_score:.2f}'
        else:
            prediction = f'NOT Spam (Safe) - Score: {combined_score:.2f}'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
