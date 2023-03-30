from optparse import Values
from statistics import mode
import streamlit as st
from joblib import load
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

stp_words=stopwords.words('english')
def clean_review(review):
  cleanreview=" ".join(word for word in review.split() if word not in stp_words)
  return cleanreview

st.cache()
def load_data():
    stp_words=stopwords.words('english')
    df=pd.read_csv('AmazonReview.csv') 
    df['Review']=df['Review'].apply(clean_review)
    df1 = df.loc[df['Sentiment']<=3,'Sentiment'] = 0
    df2 = df.loc[df['Sentiment']>3,'Sentiment'] = 1
    df['Sentiment'].value_counts()
   
    return df

def load_model():
    return load('sentiment_prediction_model.pk')

    
st.set_page_config(
    page_title="Amazon Product Reviews Sentiment Analysis",
    layout='centered',
    page_icon="🛃"
)

st.subheader("Analysis or Predict")
option = st.selectbox('**Choose a page to view**',('Analysis','Predict'))

if option == 'Analysis':
    df = load_data()
    options1 = ['View data', 'View Analysis', 'View Visualization']

    st.header('Amazon Review Sentiment Analysis')
    st.text("")
    ch = st.selectbox("select an option", options1)

    if ch == options1[0]:
       st.dataframe(df)

    if ch == options1[1]:
       st.subheader("Overall Sentiment Analysis of the Reviews")
       
       st.text("0 for Negative Sentiment")
       st.text("1 for Positive Sentiment")

    if ch == options1[2]:
       st.subheader("Graphical representation of sentiment of reviews")
       fig = px.pie(df, names='Sentiment', title='Graphical Representation of Sentiment of Reviews')
       fig.update_layout(title_x=0.5)
       st.plotly_chart(fig, use_container_width=True)

       st.subheader("Word Cloud for Negative Reviews")
       st.image('output1.png', use_column_width=True)
       st.header("Word Cloud for Positive Reviews")
       st.image('output2.png', use_column_width=True)
