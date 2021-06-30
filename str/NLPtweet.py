import re
import string
import nltk
import pandas as pd
import numpy as np
import numba as nb
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
from collections import Counter
import eng_words as ew
import number_string as ns
import time_expresion as tex
import web_comp as wc


contractions = re.compile('|'.join(ew.contraction_list))
stop_words = ew.stopwords + ns.number + tex.times
hastag = wc.hastag
tag = wc.tag
url = wc.url
emoji = wc.emoji
web_code =  wc.web_error 
punctuation = ew.punctuation

def clean(tweet):   
  # Remove url
  tweet = re.sub(url, "", tweet) 
  # Remove special characters
  tweet = re.sub(web_code, "", tweet)

  # Expand Contractions
  tweet = re.sub(r"he's", "he is", tweet)
  tweet = re.sub(r"there's", "there is", tweet)
  tweet = re.sub(r"We're", "We are", tweet)
  tweet = re.sub(r"That's", "That is", tweet)
  tweet = re.sub(r"won't", "will not", tweet)
  tweet = re.sub(r"they're", "they are", tweet)
  tweet = re.sub(r"Can't", "Cannot", tweet)
  tweet = re.sub(r"wasn't", "was not", tweet)
  tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
  tweet = re.sub(r"aren't", "are not", tweet)
  tweet = re.sub(r"isn't", "is not", tweet)
  tweet = re.sub(r"What's", "What is", tweet)
  tweet = re.sub(r"haven't", "have not", tweet)
  tweet = re.sub(r"hasn't", "has not", tweet)
  tweet = re.sub(r"There's", "There is", tweet)
  tweet = re.sub(r"He's", "He is", tweet)
  tweet = re.sub(r"It's", "It is", tweet)
  tweet = re.sub(r"You're", "You are", tweet)
  tweet = re.sub(r"I'M", "I am", tweet)
  tweet = re.sub(r"shouldn't", "should not", tweet)
  tweet = re.sub(r"wouldn't", "would not", tweet)
  tweet = re.sub(r"i'm", "I am", tweet)
  tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
  tweet = re.sub(r"I'm", "I am", tweet)
  tweet = re.sub(r"Isn't", "is not", tweet)
  tweet = re.sub(r"Here's", "Here is", tweet)
  tweet = re.sub(r"you've", "you have", tweet)
  tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
  tweet = re.sub(r"we're", "we are", tweet)
  tweet = re.sub(r"what's", "what is", tweet)
  tweet = re.sub(r"couldn't", "could not", tweet)
  tweet = re.sub(r"we've", "we have", tweet)
  tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
  tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
  tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
  tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
  tweet = re.sub(r"who's", "who is", tweet)
  tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
  tweet = re.sub(r"y'all", "you all", tweet)
  tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
  tweet = re.sub(r"would've", "would have", tweet)
  tweet = re.sub(r"it'll", "it will", tweet)
  tweet = re.sub(r"we'll", "we will", tweet)
  tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
  tweet = re.sub(r"We've", "We have", tweet)
  tweet = re.sub(r"he'll", "he will", tweet)
  tweet = re.sub(r"Y'all", "You all", tweet)
  tweet = re.sub(r"Weren't", "Were not", tweet)
  tweet = re.sub(r"Didn't", "Did not", tweet)
  tweet = re.sub(r"they'll", "they will", tweet)
  tweet = re.sub(r"they'd", "they would", tweet)
  tweet = re.sub(r"DON'T", "DO NOT", tweet)
  tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
  tweet = re.sub(r"they've", "they have", tweet)
  tweet = re.sub(r"i'd", "I would", tweet)
  tweet = re.sub(r"should've", "should have", tweet)
  tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
  tweet = re.sub(r"where's", "where is", tweet)
  tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
  tweet = re.sub(r"we'd", "we would", tweet)
  tweet = re.sub(r"i'll", "I will", tweet)
  tweet = re.sub(r"weren't", "were not", tweet)
  tweet = re.sub(r"They're", "They are", tweet)
  tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
  tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
  tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
  tweet = re.sub(r"let's", "let us", tweet)
  tweet = re.sub(r"it's", "it is", tweet)
  tweet = re.sub(r"can't", "cannot", tweet)
  tweet = re.sub(r"don't", "do not", tweet)
  tweet = re.sub(r"you're", "you are", tweet)
  tweet = re.sub(r"i've", "I have", tweet)
  tweet = re.sub(r"that's", "that is", tweet)
  tweet = re.sub(r"i'll", "I will", tweet)
  tweet = re.sub(r"doesn't", "does not", tweet)
  tweet = re.sub(r"i'd", "I would", tweet)
  tweet = re.sub(r"didn't", "did not", tweet)
  tweet = re.sub(r"ain't", "am not", tweet)
  tweet = re.sub(r"you'll", "you will", tweet)
  tweet = re.sub(r"I've", "I have", tweet)
  tweet = re.sub(r"Don't", "do not", tweet)
  tweet = re.sub(r"I'll", "I will", tweet)
  tweet = re.sub(r"I'd", "I would", tweet)
  tweet = re.sub(r"Let's", "Let us", tweet)
  tweet = re.sub(r"you'd", "You would", tweet)
  tweet = re.sub(r"It's", "It is", tweet)
  tweet = re.sub(r"Ain't", "am not", tweet)
  tweet = re.sub(r"Haven't", "Have not", tweet)
  tweet = re.sub(r"Could've", "Could have", tweet)
  tweet = re.sub(r"youve", "you have", tweet)  
  tweet = re.sub(r"donå«t", "do not", tweet)
     
  #Remove punctuation
  nopunc = [char for char in tweet if char not in string.punctuation]
  tweet  = ''.join(nopunc)
  # Convert to lower case
  tweet = tweet.lower()  
  #Remove numbers
  tweet = ''.join([i for i in tweet if not i.isdigit()])  
 
  return tweet


def clean2(tweet): 
  global stop_words, url, tag, emoji, web_code, contractions, punctuation

  # Remove url for string 'url'
  tweet = re.sub(url, "", tweet)
  # Remove tag
  tweet = re.sub(tag,"", tweet)
  # Remove emoji
  tweet = re.sub(emoji,"", tweet)
  # Remove special characters
  tweet = re.sub(web_code, "", tweet)  
  # Convert to lower case
  tweet = tweet.lower()      
  # Remove Contractions
  tweet = re.sub(contractions, " ", tweet)
  # organice string
  for i in punctuation:
    tweet = " ".join(tweet.split(i))    
  # Remove punctuation
  nopunc = [char for char in tweet if char not in string.punctuation]
  tweet  = ''.join(nopunc)
  # Remove stopwords 
  tweet =[char for char in tweet.split() if char not in stop_words]
  tweet  = " ".join(tweet)
  #Remove numbers
  tweet = ''.join([i for i in tweet if not i.isdigit()])  
   
  return tweet

def extract_web_comp(tweet):
  wc = hastag.findall(tweet.lower())
  wc = wc + tag.findall(tweet.lower())
  wc = wc + re.findall("http",tweet)
  wc = wc + emoji.findall(tweet)
  return " ".join(wc) 

def remove(x,y):
  k = [char for char in x.split() if char not in y]
  x  = " ".join(k)
  return x

def concat(df, var, sep = ' '):
   text = ''
   text += ''.join(i for i in df[var])+sep
   return text

def concat_list(df, var):
  ht = []
  [[ht.append(j) for j in i]  for i in df[var]]
  return ht

def corpus(df,var):
  corp = []
  [corp.append(i) for i in df[var]]
  return corp

def Compare_most_common(list0,list1,n):
  list0 = Counter(list0)
  list1 = Counter(list1)
  list0=list0.most_common(n)
  list1=list1.most_common(n)
  res_list0=[x[0] for x in list0]
  res_list1=[x[0] for x in list1]
  text = []
  for item in res_list0:
    if item in res_list1:
      text.append(item)
  return text

def word_cloud(text):
  comment_words = '' 
  stopwords = set(STOPWORDS) 
  # split the value 
  tokens = text.split()
  comment_words += " ".join(tokens)+" "
  
  wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)
  return wordcloud 


def plot_wordsclouds(t1,t2):

  wordcloud1 = word_cloud(t1)
  wordcloud2 = word_cloud(t2)

  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.set_figheight(8)
  fig.set_figwidth(16)
  fig.suptitle('Compare words frecuencies')
  ax1.set_title('Real Disaster')
  ax1.imshow(wordcloud1) 
  ax1.axis("off") 
  ax2.set_title('Not Disaster')
  ax2.imshow(wordcloud2)
  ax2.axis("off")

def param(estimator):
  if estimator.__class__.__name__ == 'RandomForestClassifier':
    p1 = 'Max depth: ' + str(estimator.max_depth)
  elif estimator.__class__.__name__ == 'SVC':
    p1 = 'Gamma: ' + str(estimator.gamma)
  return p1

def learning_data(enc, clean, data, estimators, X, y, test_size=None, 
                  scoring=None, iters=1, train_sizes=None):

  from sklearn.model_selection import train_test_split
  for estimator in estimators:
    for i in test_size:
      m_tr = []
      m_ts = []
      for iter in range(iters):    
        Xtr, Xts, ytr, yts = train_test_split(X,y, test_size=i)
        estimator.fit(Xtr,ytr)
        m_tr.append(scoring(Xtr, ytr, estimator = estimator))
        m_ts.append(scoring(Xts, yts, estimator = estimator))
      p1 = param(estimator)
      data['model'].append(estimator.__class__.__name__)
      data['estimator'].append(estimator)
      data['train'].append(1-i)
      data['mean_tr'].append(np.mean(m_tr))
      data['sd_tr'].append(np.std(m_tr))
      data['mean_ts'].append(np.mean(m_ts))
      data['sd_ts'].append(np.std(m_ts))
      data['p1'].append(p1)
      data['encodding'].append(enc)
      data['clean'].append(clean)
  return data

def learning_dt(enc, clean, data, estimators, X, y, yc, test_size=None, scoring=None, iters = 1, train_sizes=None):

  from sklearn.model_selection import train_test_split
  for estimator in estimators:
    for i in test_size:
      m_tr = []
      m_ts = []
      for iter in range(iters): 
        yr = np.array([])
        ys = yr
        yrp = yr
        ysp = yr
        for c in np.unique(yc):
          l = yc == c
          xi = X[l]
          yi = y[l]
          Xtr, Xts, ytr, yts = train_test_split(xi,yi, test_size=i)
          estimator.fit(Xtr,ytr)
          yr = np.concatenate((yr,ytr))
          yrp = np.concatenate((yrp ,estimator.predict(Xtr)))
          ys = np.concatenate((ys, yts))
          ysp = np.concatenate((ysp, estimator.predict(Xts)))
        
        m_tr.append(scoring(X = z, y = yr, ypred = np.array(yrp)))
        m_ts.append(scoring(X = z, y = np.array(ys), ypred = np.array(ysp)))
      p1 = param(estimator)
      data['model'].append(estimator.__class__.__name__)
      data['estimator'].append(estimator)
      data['train'].append(1-i)
      data['mean_tr'].append(np.mean(m_tr))
      data['sd_tr'].append(np.std(m_tr))
      data['mean_ts'].append(np.mean(m_ts))
      data['sd_ts'].append(np.std(m_ts))
      data['p1'].append(p1)
      data['encodding'].append(enc)
      data['clean'].append(clean)
  return data

def plot_size_efect(df):
  
  models = list(df['model'].value_counts().index)
  for i in models:
    d = df[ df['model'] == i]
    pm1 = list(d['p1'].value_counts().index)
    for j in pm1:
      dj = d[d['p1'] == j]
      plt.ylim((0.0, 1.0)) 
      plt.suptitle(i, fontsize=18)
      plt.title(j, fontsize=10)
      plt.xlabel("Training sample")
      plt.ylabel("Score")
      plt.grid()

      tr = dj['train']
      mtr= dj['mean_tr']
      sdtr = dj['sd_tr']
      mts= dj['mean_ts']
      sdts = dj['sd_ts']

      plt.fill_between(tr, mtr - sdtr,
                   mtr + sdtr, alpha=0.1,
                   color="r")
      plt.fill_between(tr, mts - sdts,
                   mts + sdts, alpha=0.1,
                   color="g")
      plt.plot(tr, mtr, 'o-', color="r",
               label="train score")
      plt.plot(tr, mts, 'o-', color="g",
               label="test score")
      plt.legend(loc="best")
      plt.show()
      plt.close()

def plot_parameter_efect(df):
  
  models = list(df['model'].value_counts().index)
  for i in models:
    d = df[ df['model'] == i]
    trsz = list(d['train'].value_counts().index)
    for j in trsz:
      dj = d[d['train'] == j]
      plt.ylim((0.0, 1.0)) 
      plt.suptitle(i, fontsize=18)
      plt.title('train size: '+ str(j), fontsize=10)
      plt.xlabel("Parameter value")
      plt.ylabel("Score")
      plt.grid()


      dj = d[d['train'] == j]      
      pm1 = dj['p1']
      mtr= dj['mean_tr']
      sdtr = dj['sd_tr']
      mts= dj['mean_ts']
      sdts = dj['sd_ts']

      plt.fill_between(pm1, mtr - sdtr,
                   mtr + sdtr, alpha=0.1,
                   color="r")
      plt.fill_between(pm1, mts - sdts,
                   mts + sdts, alpha=0.1,
                   color="g")
      plt.plot(pm1, mtr, 'o-', color="r",
               label="train score")
      plt.plot(pm1, mts, 'o-', color="g",
               label="test score")
      plt.legend(loc="best")
      plt.show()
      plt.close()


