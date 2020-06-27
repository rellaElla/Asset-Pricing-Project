import tweepy
import os
import time
import xlwt
from xlwt import Workbook

consumer_key = os.environ['twitter_consumer_key']
consumer_secret = os.environ['twitter_consumer_secret']
access_key = os.environ['twitter_access_key']
access_secret = os.environ['twitter_access_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def get_tweets(ticker, count = 1000):

    ret = []
    tweets = []

    q=str(ticker)
    fetched_tweets = api.search(q, count = count, lang="en")

    for tweet in fetched_tweets:
        if(tweet.text not in tweets):
            tweets.append(tweet.text)
            t = (ticker, tweet.created_at, tweet.user.screen_name, tweet.text)
            ret.append(t)

    return ret

def main():

    ticker = "$XOM"
    tweets = get_tweets(ticker)
    l = len(tweets)

    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Ticker')
    sheet1.write(0, 1, 'Date of Tweet')
    sheet1.write(0, 2, 'User')
    sheet1.write(0, 3, 'Tweet Content')

    j = 1
    for i in tweets:
        sheet1.write(j, 0, i[0])
        sheet1.write(j, 1, i[1])
        sheet1.write(j, 2, i[2])
        sheet1.write(j, 3, i[3])
        print(i[1])
        j+=1

    wb.save('output.xls')


if(__name__ == "__main__"):
    main()
