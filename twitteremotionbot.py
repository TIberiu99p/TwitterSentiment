# -*- coding: utf-8 -*-
"""twitterEmotionBot.ipynb

Coded in  Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15crQVSEIgg6vLyKCLcxtEpE7qW_awPJ5
"""

#libraries imported
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#keys
consumerKey = 'A6ebS1uRqjNHaXSebTFwBi94k'
consumerSecret = 'bXq1Q2zZ57tIsLbJY0nxNRL8nlkbP5X9LJ2Gto5nmBfZoSldqi'
bearerToken = 'AAAAAAAAAAAAAAAAAAAAAH8QlAEAAAAA0wrxNvZRxNcSJkrMWRbzqwU1Dks%3Dq8emiSJGuRDDeHlhiMCkfC22aYzuRpQQams6POG1UFPAzOTkNy'
accessToken = '1174703588565757953-j78skCtMlKVerQUbMbMyTzCz9JNSD6'
acessTokenSecret = 'vfGciWh0hYBYcARqzFjbg44d9L9IMEKrxnIsZqNztQJYO'

#authenticate obj created
authenticate = tweepy.OAuthHandler(consumerKey,consumerSecret)
#set access token and access token secret
authenticate.set_access_token(accessToken,acessTokenSecret)
#create api obj while passing auth information 
api = tweepy.API(authenticate, wait_on_rate_limit = True)

#Gather 1500 tweets about tesla earnings 
#create search term
search_term = 'One Piece Red Film'
#create cursor obj
tweets = tweepy.Cursor(api.search,q = search_term,lang='en',since='2022-01-01',tweet_mode='extended').items(1500)

#store tweets 
all_tweets = [tweet.full_text for tweet in tweets]
all_tweets

#data frame to store the tweets 
df = pd.DataFrame(all_tweets, columns=['Tweets'])

#first 6 rows of data 
df.head(6)

#Data cleaning
def cleanTwt(twt):
  twt = re.sub('RT', ' ',twt) #remove rt
  twt = re.sub('#[A-Za-z0-9]+', ' ',twt)#remove the `#` from tweets
  twt = re.sub('\\n', '', twt)#remove new slash characters
  twt = re.sub('https?:\/\/\S+', ' ',twt)#remove hyperlinks
  twt = re.sub('@[\S]*', ' ',twt)#remove mentions
  twt = re.sub('[\s]+|[\s]+$', ' ', twt)#remove leading and whitespaces
  return twt

#new column with clean tweets
df['Clean_tweets'] = df['Tweets'].apply(cleanTwt)
#show data
df.head()

df = pd.DataFrame(df['Clean_tweets'],columns=['Clean_tweets'])
#remove duplicates row
df.drop_duplicates(inplace=True)
idx = list(range(0,len(df)))
df = df.set_index(pd.Index(idx))
#show data
df

#function for subjectivity
def getSubjectivity(twt):
  return TextBlob(twt).sentiment.subjectivity

#function to get polarity
def getPolarity(twt):
  return TextBlob(twt).sentiment.polarity

#two new columns for our data set for subjectivity and polarity
df['Subjectivity'] = df['Clean_tweets'].apply(getSubjectivity)
df['Polarity'] = df['Clean_tweets'].apply(getPolarity)
#Show data
df.head(6)

#classify positivity > 0, negativity < 0, neutrail = 0
def getSentiment(value):
  if value < 0:
    return 'Negative'
  elif value > 0:
    return 'Positive'
  else:
    return 'Neutral'

#create column Sentiment
df['Sentiment'] = df['Polarity'].apply(getSentiment)
df.head()

#create a scatter plot
plt.figure(figsize = (8,6))
for i in range(0, len(df)):
  plt.scatter(df['Polarity'][i],df['Subjectivity'][i],color = 'green')
plt.title('Scatter Plot')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

#create bar chart
df['Sentiment'].value_counts().plot(kind='bar')
plt.title('Bar Plot')
plt.xlabel('Sentiment')
plt.ylabel('Number of tweets')
plt.show()