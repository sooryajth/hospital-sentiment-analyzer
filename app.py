import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)

# Stopwords with negation preserved
stop_words = set(stopwords.words('english'))
negation_words = {'no', 'not', 'nor', "don't", "didn't",
                  "isn't", "wasn't", "aren't", "won't", "can't"}
stop_words = stop_words - negation_words

# Load saved model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning function — must match training cleaning exactly
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split()
                     if word not in stop_words])
    return text

# App title
st.title('Hospital Review Sentiment Analyzer')

st.write('---')

# Single review prediction
st.subheader('Analyze a Patient Review')
review = st.text_area('Enter patient review here')

if st.button('Analyze Sentiment'):
    if review.strip() == '':
        st.warning('Please enter a review first')
    elif len(review.split()) < 3:
        st.warning('Review too short. Please enter at least 3 words.')
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        if prediction == 0:
            conf_score = round(confidence[0] * 100, 1)
            if conf_score < 60:
                st.warning('UNCERTAIN — Needs Human Review')
                st.write(f'Confidence too low: {conf_score}%')
                st.write('Model is not confident enough — manual review recommended')
            else:
                st.error('NEGATIVE REVIEW')
                st.write(f'Confidence: {conf_score}%')
                st.write('This patient needs immediate follow up')
        else:
            conf_score = round(confidence[1] * 100, 1)
            if conf_score < 60:
                st.warning('UNCERTAIN — Needs Human Review')
                st.write(f'Confidence too low: {conf_score}%')
                st.write('Model is not confident enough — manual review recommended')
            else:
                st.success('POSITIVE REVIEW')
                st.write(f'Confidence: {conf_score}%')
                st.write('Patient had a satisfying experience')

st.write('---')

# Business insights section
st.subheader('Key Business Insights')

col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Total Reviews', '996')
    st.metric('Positive', '728 (73%)')

with col2:
    st.metric('Negative', '268 (27%)')
    st.metric('Model Accuracy', '88%')

with col3:
    st.metric('Complaint Recall', '88%')
    st.metric('Complaints Caught', '51/58')

st.write('---')

# Top complaint drivers

st.write('---')

#