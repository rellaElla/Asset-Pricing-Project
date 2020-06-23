import tweepy
import os

consumer_key = os.environ['twitter_consumer_key']
consumer_secret = os.environ['twitter_consumer_secret']
access_key = os.environ['twitter_access_key']
access_secret = os.environ['twitter_access_secret']


def get_tweets(username):

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)

        tweets = api.user_timeline(screen_name=username, count = 1)

        tmp=[]

        tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created

        for j in tweets_for_csv:
            tmp.append(j)
            #print(j+"\n")
        return tmp


def main():
    # TO-DO: format tweets - get_tweets should return an array of tuples
    # composed of (stock ticker, tweet, date of tweet, user)
    # store each tuple in a csv
    get_tweets("realDonaldTrump")
    get_tweets("wallstreetbets")
    get_tweets("Benzinga")
    get_tweets("Stocktwits")
    get_tweets("CNBC")



if(__name__ == "__main__"):
    main()
