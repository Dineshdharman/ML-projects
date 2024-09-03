import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import streamlit as st

# Set up the Streamlit app
st.title("Twitter Sentiment Analysis")

# Load the dataset
df = pd.read_csv("./Twitter_Data.csv")
nltk.download('stopwords')


# Inspect the dataset
st.header("Dataset Overview")
st.write(df.head())
st.write(df.info())
st.write("Missing values in each column:", df.isnull().sum())

# Data cleaning and preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df = df.dropna(subset=['clean_text'])
df['clean_text'] = df['clean_text'].astype(str)

df['cleaned_review'] = df['clean_text'].apply(preprocess_text)

# Univariate analysis
st.header("Univariate Analysis")
st.subheader("Category Distribution")
sns.countplot(data=df, x='category')
st.pyplot(plt)  # Render the plot for category distribution

# Bivariate analysis
st.header("Bivariate Analysis")
st.subheader("Review Length by Category")
df['review_length'] = df['cleaned_review'].apply(len)

sns.boxplot(x='category', y='review_length', data=df)
st.pyplot(plt)  # Render the plot for review length by category

st.write("Mean review length by category:", df.groupby('category')['review_length'].mean())

# Wordcloud analysis
st.header("Wordcloud Analysis")
positive_review = ' '.join(df[df['category'] == 1]['cleaned_review'])
negative_review = ' '.join(df[df['category'] == -1]['cleaned_review'])
wordcloud_positive = WordCloud().generate(positive_review)
wordcloud_negative = WordCloud().generate(negative_review)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(wordcloud_positive, interpolation='bilinear')
ax[0].axis('off')
ax[0].set_title('Positive Reviews')

ax[1].imshow(wordcloud_negative)
ax[1].axis('off')
ax[1].set_title('Negative Reviews')

st.pyplot(fig)  # Render the wordclouds

# Text data visualization
st.header("Text Data Visualization")
positive_words = Counter(' '.join(df[df['category'] == 1]['cleaned_review']).split())
negative_words = Counter(' '.join(df[df['category'] == -1]['cleaned_review']).split())

positive_review_df = pd.DataFrame(positive_words.most_common(9), columns=['words', 'count'])
negative_review_df = pd.DataFrame(negative_words.most_common(5), columns=['words', 'count'])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x='words', y='count', data=positive_review_df, ax=ax[0])
ax[0].set_title('Most Common Words in Positive Reviews')

sns.barplot(x='words', y='count', data=negative_review_df, ax=ax[1])
ax[1].set_title('Most Common Words in Negative Reviews')

st.pyplot(fig)  # Render the bar plots for most common words
