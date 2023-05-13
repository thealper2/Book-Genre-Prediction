import streamlit as st
import pickle
import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

model = pickle.load(open("SVCmodel.pickle", "rb"))
tfidf = pickle.load(open("TFIDFmodel.pickle", "rb"))

stop_words_list = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
	text = text.lower()
	text = re.sub(r"\'", " ", text)
	text = text.translate(str.maketrans("", "", string.punctuation))
	words = text.split()
	words = [word for word in words if not word in stop_words_list]
	words = [word.strip() for word in words if len(word.strip()) > 1]
	text = " ".join(words)
	return text


def lemmatizing(text):
	stemmed_text = ""
	for word in text.split():
		stem = lemmatizer.lemmatize(word)
		stemmed_text += stem
		stemmed_text += " "

	stemmed_text = stemmed_text.strip()
	return stemmed_text

def stemming(text):
	stemmed_text = ""
	for word in text.split():
		stem = stemmer.stem(word)
		stemmed_text += stem
		stemmed_text += " "

	stemmed_text = stemmed_text.strip()
	return stemmed_text

def genre_classificate(result):
	if result == 0:
		return st.success("Crime")
	elif result == 1:
		return st.success("Fantasy")
	elif result == 2:
		return st.success("History")
	elif result == 3:
		return st.success("Horror")
	elif result == 4:
		return st.success("Psychology")
	elif result == 5:
		return st.success("Romance")
	elif result == 6:
		return st.success("Science")
	elif result == 7:
		return st.success("Sports")
	elif result == 8:
		return st.success("Thriller")
	else:
		return st.success("Travel")

st.title("Book Genre Classification")

text = st.text_area("Write book summary")

if st.button("Predict"):
	text = preprocess(text)
	text = lemmatizing(text)
	text = stemming(text)
	tfidf_text = tfidf.transform([text])
	result = model.predict(tfidf_text).item()
	print(result)
	genre_classificate(result)
