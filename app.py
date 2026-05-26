
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning function
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
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]
        
        if prediction == 0:
            st.error('NEGATIVE REVIEW')
            st.write(f'Confidence: {round(confidence[0]*100, 1)}%')
            st.write('This patient needs immediate follow up')
        else:
            st.success('POSITIVE REVIEW')
            st.write(f'Confidence: {round(confidence[1]*100, 1)}%')
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
    st.metric('Model Accuracy', '85%')

with col3:
    st.metric('Complaint Recall', '90%')
    st.metric('Complaints Caught', '52/58')

st.write('---')
st.subheader('Top Complaint Drivers')
st.write('1. Waiting times — emergency and lab departments')
st.write('2. Billing transparency issues')
st.write('3. Service quality in specific departments')
