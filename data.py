import os
import pandas as pd
import numpy as np
import emoji
import wordsegment
from config import OLID_PATH
from utils import pad_sents, get_mask, get_lens

wordsegment.load()

def read_df(filepath: str):
    df = pd.read_csv( filepath , keep_default_na=False)

    tweets = np.array(df['tweet'].values)

    # Process tweets
    tweets = process_tweets(tweets)

    label_a = np.array( df['sentiment_score'].values)
    label_b = np.array( df['annotator_score'].values )
    label_c = np.array( df['directness_score'].values)
    label_d = np.array( df['group_score'].values )
    label_e = np.array( df['target_score'].values )
    
    nums = len(df)

    return nums , tweets , label_a, label_b, label_c , label_d , label_e

def process_tweets(tweets):
    # Process tweets
    tweets = emoji2word(tweets)
    tweets = replace_rare_words(tweets)
    tweets = remove_replicates(tweets)
    tweets = segment_hashtag(tweets)
    tweets = remove_useless_punctuation(tweets)
    tweets = np.array(tweets)
    return tweets

def emoji2word(sents):
    return [emoji.demojize(sent) for sent in sents]

def remove_useless_punctuation(sents):
    for i, sent in enumerate(sents):
        sent = sent.replace(':', ' ')
        sent = sent.replace('_', ' ')
        sent = sent.replace('...', ' ')
        sents[i] = sent
    return sents

def remove_replicates(sents):
    # if there are multiple `@USER` tokens in a tweet, replace it with `@USERS`
    # because some tweets contain so many `@USER` which may cause redundant
    for i, sent in enumerate(sents):
        if sent.find('@USER') != sent.rfind('@USER'):
            sents[i] = sent.replace('@USER', '')
            sents[i] = '@USERS ' + sents[i]
        
    return sents

def replace_rare_words(sents):
    rare_words = {
        '@URL': 'http'
    }
    for i, sent in enumerate(sents):
        for w in rare_words.keys():
            sents[i] = sent.replace(w, '' )
    return sents

def segment_hashtag(sents):
    # E.g. '#LunaticLeft' => 'lunatic left'
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(' ')
        for j, t in enumerate(sent_tokens):
            if t.find('#') == 0:
                sent_tokens[j] = ' '.join(wordsegment.segment(t))
        sents[i] = ' '.join(sent_tokens)
    return sents

def all_tasks(filepath: str, tokenizer, truncate=512):
    # read the dataset 
    nums , tweets , label_a, label_b, label_c , label_d , label_e = read_df(filepath)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [ tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums) ]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return  token_ids, lens, mask, label_a, label_b, label_c , label_d , label_e

