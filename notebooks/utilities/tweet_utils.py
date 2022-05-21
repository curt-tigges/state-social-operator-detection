import pandas as pd
import re

def clean_text(text):
    """Cleans up Tweet text
    
    Args:
        text (string): Text to be cleaned

    Returns:
        text (string): cleaned-up text
    
    """
    text = re.sub(r"http\S+", "", text)
    # remove newlines and carriage returns
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\r", " ", text)
    text = re.sub(r"\'t", " not", text) # Change 't to 'not'
    # remove @ mentions, digits, and special characters
    text = re.sub(r'(@.*?)[\s]', ' ', text) # @
    text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", text) # digits
    text = re.sub(r"[^\w\s]", "", text) # special characters
    # clean up spaces
    text = text.strip(" ")
    text = re.sub(' +',' ', text).strip()

    return text

def combine_tweets(dataframe, tweet_count):
    """Given a specific Tweet, combines with the last N Tweets in a single string
    
    Args:
        dataframe (pandas dataframe): Cleaned Tweets with metadata
        tweet_count (int): Number of Tweets (N) to combine

    Returns:
        pandas dataframe: Original dataframe, plus combined N Tweet string
    """
    dataframe = dataframe.sort_values(['userid','tweet_time'], ascending=[True,False])
    
    cols = ['userid','tweet_time','clean_tweets']
    tmp_df = dataframe[cols].copy()
    tmp_df = tmp_df.sort_values(['userid','tweet_time'], ascending=[True,False])
    
    tweet_sequences = []

    for i in range(tmp_df.shape[0]):
        id = tmp_df.iloc[i, 0]
        last_tweets = ""

        for j in range(i,i+tweet_count):
            if j < tmp_df.shape[0] and tmp_df.iloc[j, 0] == id:
                last_tweets = last_tweets + tmp_df['clean_tweets'].iloc[j] + " | "

        tweet_sequences.append(last_tweets)
    
    dataframe['recent_tweets'] = tweet_sequences
    
    return dataframe

def apply_filters(df):
    """Filters and cleans dataframe of tweets
    
    Args:
        df (dataframe): Incoming dataframe with 'tweet_text' column

    Returns:
        df (dataframe): Filtered dataframe
    
    """
    df["clean_tweets"] = (df["tweet_text"].map(lambda text: clean_text(text)))
    df['word_count'] = df['clean_tweets'].str.count(' ') + 1
    crit1 = ~df["tweet_text"].str.startswith("RT")
    crit2 = ~df["clean_tweets"].isnull()
    crit3 = df["clean_tweets"] != ""
    crit4 = df["word_count"] > 3

    df = df[crit1 & crit2 & crit3 & crit4].copy()

    return df