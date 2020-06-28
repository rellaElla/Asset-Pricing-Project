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

def analyze(tickers):
    ret = []
    for s in tickers:
        ret.append(get_tweets(s))
    return ret

def get_tweets(ticker, count = 1000):

    ret = []
    tweets = []

    q=str(ticker)
    # fetched_tweets = api.search(q, count = count, lang="en")
    fetched_tweets = tweepy.Cursor(api.search, q="$XOM", count = 100, lang="en", tweet_mode='extended').items()

    for tweet in fetched_tweets:
        if(tweet.full_text not in tweets): # filtering out duplicate tweets
            tweets.append(tweet.full_text)
            date = str(tweet.created_at)
            t = (ticker, date, tweet.user.screen_name, tweet.full_text, tweet.retweet_count)
            ret.append(t)

    return ret

def write(tweets):

    wb = Workbook()

    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Ticker')
    sheet1.write(0, 1, 'Date of Tweet')
    sheet1.write(0, 2, 'User')
    sheet1.write(0, 3, 'Tweet Content')
    sheet1.write(0, 4, 'Number of Retweets')

    k=1

    for i in tweets:
        for j in i:
            sheet1.write(k, 0, j[0]) # Ticker - stock being tweeted about
            sheet1.write(k, 1, j[1]) # date of tweet
            sheet1.write(k, 2, j[2]) # User - person who tweeted
            sheet1.write(k, 3, j[3]) # Tweet content
            sheet1.write(k, 4, j[4]) # number of times that the tweet has been retweeted
            k+=1

    wb.save('output.xls') # writing data to output file

def main():

    tickers = ["$XOM","$SNAP","$BA","$AAPL","$FB"] # companies to analyze
    tweets = analyze(tickers) # gathering tweets about the companies
    write(tweets) # writing data to .xls file

if(__name__ == "__main__"):
    main()
