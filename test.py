import tweepy
import os
import time
import pandas as pd
import xlwt
from xlwt import Workbook

consumer_key = os.environ['twitter_consumer_key']
consumer_secret = os.environ['twitter_consumer_secret']
access_key = os.environ['twitter_access_key']
access_secret = os.environ['twitter_access_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

tweets = tweepy.Cursor(api.search, q="$XOM", count = 100, lang="en", tweet_mode='extended').items()
for tweet in tweets:
    content = tweet.full_text
    date = tweet.created_at
    print(date, ":",content,"\n\n")

print("testing")
