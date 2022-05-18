# import tweepy
import tweepy

# api keys from twitter developers
consumer_key = '<your consumer_key here>'
consumer_secret = '<your consumer_secret here>'
access_token = '<your access_token here>'
acess_token_sectret = '<your acess_token_sectret here>'

# login to twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, acess_token_sectret)
auth.secure = True
api = tweepy.API(auth)

# define the id of the tweet you are looking for
id1 = 1181329749966295041
id2 = 572348008882302976

# get the tweet
tweet = api.get_status(id1)
tweet2 = api.get_status(id2)
# print the text of the tweet
print("First tweet")
print(tweet)
print(tweet.text)
print("\n\n")

print("Second tweet")
print(tweet2)
print(type(tweet2))
print(tweet2.text)
