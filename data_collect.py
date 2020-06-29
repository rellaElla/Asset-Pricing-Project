import tweepy
import os
import time
import pandas as pd
import xlwt
from xlwt import Workbook
import datetime

consumer_key = os.environ['twitter_consumer_key']
consumer_secret = os.environ['twitter_consumer_secret']
access_key = os.environ['twitter_access_key']
access_secret = os.environ['twitter_access_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(60)

def analyze(tickers, sources):
    ret = []
    for s in tickers:
        for source in sources:
            ret.append(get_tweets2(s, source))
    return ret

def get_tweets(ticker):

    ret = []
    tweets = []

    q=str(ticker)

    fetched_tweets = api.search(q="$XOM", lang="en", tweet_mode='extended', count=10000)
    tweets.extend(fetched_tweets)
    oldest_id = fetched_tweets[-1].id

    while True:
        t = api.search(q="$XOM", lang="en", tweet_mode='extended', count=10000, max_id=oldest_id-1)
        if len(t) == 0:
            break
        oldest_id = t[-1].id
        tweets.extend(t)
    for tweet in tweets:
        date = str(tweet.created_at)
        t = (ticker, date, tweet.user.screen_name, tweet.full_text, tweet.retweet_count)
        ret.append(t)


    return ret

def get_tweets2(ticker, userID):

    tweets = api.user_timeline(screen_name=userID,count=10000, q=ticker, tweet_mode = 'extended')
    all_tweets = []
    ret = []
    all_tweets.extend(tweets)
    oldest_id = tweets[-1].id
    while True:
        tweets = api.user_timeline(screen_name=userID,
                               count=10000,
                               max_id = oldest_id - 1,
                               q=ticker,
                               tweet_mode = 'extended'
                               )
        if len(tweets) == 0:
            break
        oldest_id = tweets[-1].id
        all_tweets.extend(tweets)

    for tweet in all_tweets:
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

    # tickers = ["$XOM","$SNAP","$BA","$AAPL","$FB"] # companies to analyze
    tickers = ["$XOM"]
    sources = ["CNBC", "Stocktwits", "Benzinga", "WSJmarkets"]
    #tweets = analyze(tickers) # gathering tweets about the companies
    tweets = analyze(tickers, sources)
    write(tweets) # writing data to .xls file

if(__name__ == "__main__"):
    main()
