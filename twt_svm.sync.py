# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:myenv] *
#     language: python
#     name: conda-env-myenv-py
# ---

# %% [markdown]
# ## Import statements
# %%
import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
# %% [markdown]
# ## Downloading the data and understanding the structure
# %%
nltk.download('twitter_samples')
# %%
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
# %%
# get number of positive_tweets and negative_tweets
print('Number of pos tweets: ', len(positive_tweets))
print('Number of neg tweets: ', len(negative_tweets))
# %% [markdown]
# Evidently, there are an equal number of positive and negative tweets.
# %%
print("The type of the list of tweets is:", type(positive_tweets))
print("The type of a single tweet is:", type(positive_tweets[0]))
# %% [markdown]
# View the raw tweets.
# %%
positive_tweets[:10]
# %% [markdown]
# ## Comparing the lengths of the two types of tweets
# %%
total_positive_words = []
for sentence in positive_tweets:
    total_positive_words.append(sentence.count(' '))

total_negative_words = []
for sentence in negative_tweets:
    total_negative_words.append(sentence.count(' '))

# View the results as a histogram
import plotly.graph_objects as go
import numpy as np

x0 = np.array(total_positive_words)
x1 = np.array(total_negative_words)

fig = go.Figure()
fig.add_trace(go.Histogram(x=x1, name = 'Negative'))
fig.add_trace(go.Histogram(x=x0, name = 'Positive'))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()
# %% [markdown]
# ## Preprocessing
#
# There are twitter handles, punctuations, emojis, etc. that are irrelevant to the
# sentiment of the tweet. We must remove them.
# Choosing a random tweet so we can see the changes
# %%
tweet = positive_tweets[1223]
print(tweet)
# %% [markdown]
# ### Imports
# %%
nltk.download('stopwords')

import re # for substituting using regular expressions
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # remove endings like -ing, -s, etc.
from nltk.tokenize import TweetTokenizer

# %% [markdown]
# ### Removing links, twitter marks like hashtags and styles
# %%
print("unchanged tweet:")
print(tweet)

# old retweets have an "RT" at the beginning
tweet2 = re.sub(r'^RT[\s]+', '', tweet)

# remove hyperlinks
tweet2 = re.sub(r'https?:\/\/\S*[\s\r\n]*', '', tweet2)

# remove hashtags but don't remove the words from the hashtags as they may provide
# clues about the sentiment of the tweet
tweet2 = re.sub(r'#', '', tweet2)
# # remove twitter handles 
# tweet2 = re.sub(r'@\S*[\r\n\s]*','',tweet2)

# remove emojis
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
tweet2 = remove_emoji(tweet2)

# remove single numeric terms from the tweet
tweet2 = re.sub(r'[0-9]','', tweet2)

print("\nafter removing RT, links, hashtags and numbers")
print(tweet2)

# %% [markdown]
# ### Tokenizing
# Here, we split the strings into their individual words, remove blanks, tabs and
# handles, and convert them to lowercase
# %%
print("before tokenizing")
print(tweet2)

tokenizer = TweetTokenizer(preserve_case=False,
                           strip_handles=True,
                           reduce_len=True)
# perform tokenizing
tweet_tokenized = tokenizer.tokenize(tweet2)

print('\ntokenized string;')
print(tweet_tokenized)
# %% [markdown]
# ### Stopwords and Punctuation
# Stopwords are words like "are", "in", etc. that don't contribute to the sentiment of the sentence.
# Punctuation, similarly, is irrelevant, and therefore, must be removed.
# %%
eng_stopwords = stopwords.words('english')
print("stopwords: ")
print(eng_stopwords)
print("\npunctuation: ")
print(string.punctuation)
# %% [markdown]
# Upon looking at the stopwords, we see that some may be relevant to the sentiment analysis we
# are going to perform. So we cannot remove all the stopwords from the nltk stopwords corpus.
# %%
print("before stopword and punctuation removal")
print(tweet_tokenized)

clean_twt = []
stopwords_to_keep = ['because', 'not', 'won', 'against', 'between']

for word in tweet_tokenized:
    if (word in stopwords_to_keep or (word not in eng_stopwords and word not in string.punctuation)):
        clean_twt.append(word)

print("after stopword and punctuation removal")
print(clean_twt)
# %% [markdown]
# ### Stemming
# stemming is the removal of superfluous endings to words like -ing, -ness, -hood, -er, etc. to
# obtain the stem of the related words.
# %%
stemmer = PorterStemmer()

twt_stems = []

for word in clean_twt:
    stem_wrd = stemmer.stem(word)
    twt_stems.append(stem_wrd)

print("words after stemming:")
print(twt_stems)
