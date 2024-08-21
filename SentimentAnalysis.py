import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

#load the dataset
df=pd.read_csv(r"C:\Users\HP\Downloads\Twitter_Data.csv")

#Inspect the dataset
df.head()
df.info()
df.isnull().sum()

#Data cleaning and preprocessing
stop_words=set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text=text.lower()
    text=re.sub(r'[^a-z\s]', '', text)
    text=' '.join(word for word in text.split() if word in stop_words)
    return text

df=df.dropna(subset=['clean_text'])
df['clean_text']=df['clean_text'].astype(str)

df['cleaned_review']=df['clean_text'].apply(preprocess_text)

                                       
#univariate analysis
sns.countplot(data=df,x='category')
plt.show()


#bivariate analysis
df['review_length']=df['cleaned_review'].apply(len)

sns.boxplot(x='category',y='review_length',data=df)
print(df.groupby('cleaned_review')['review_length'].mean())

#wordcloud analysis
positive_review=' '.join(df[df['category']== 1]['cleaned_review'])
negative_review=' '.join(df[df['category']== -1]['cleaned_review'])
wordcloud_positive = WordCloud().generate(positive_review)
wordcloud_negative = WordCloud().generate(negative_review)  

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews')

plt.subplot(1,2,2)
plt.imshow(wordcloud_negative)
plt.axis('off')
plt.title('Negative Reviews')

plt.show()

#text data visulization
positive_words=Counter(' '.join(df[df['category']== 1]['cleaned_review']).split())
negative_words=Counter(' '.join(df[df['category']== -1]['cleaned_review']).split())

positive_review_df=pd.DataFrame(positive_words.most_common(9), columns=['words','count'])
negative_review_df=pd.DataFrame(negative_words.most_common(5), columns=['words','count'])

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.barplot(x='words',y='count',data=positive_review_df)
plt.title('Most words in Positive Reviews')

plt.subplot(1,2,2)
sns.barplot(x='words',y='count',data=negative_review_df)
plt.title('Most words in Ngative Reviews')

plt.show()
