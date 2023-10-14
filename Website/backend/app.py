# app.py

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from spam_detector import SpamDetector
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.isalnum()]
    filtered_tokens = [token for token in filtered_tokens if token not in stopwords.words('english') and token not in string.punctuation]
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
spam_detector = SpamDetector(tfidf, model)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_spam(request: Request):
    form_data = await request.form()
    text = form_data.get("text")
    transformed_text = transform_text(text) if text else ""
    vector_input = tfidf.transform([transformed_text])
    result = spam_detector.predict(vector_input)
    prediction = "Spam" if result == 1 else "Not Spam"
    return templates.TemplateResponse(
        "index.html", {"request": request, "prediction": prediction, "text": text}
    )
