# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @File    : sample_project.py

# package import
import pandas as pd
import numpy as np
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus.graphviz import graph_from_dot_data
import graphviz
import string
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from pydotplus.graphviz import graph_from_dot_data
from sklearn.model_selection import cross_validate
import re

import requests
import newspaper
from newspaper import Article
from newsapi import NewsApiClient # Needed for using API Feed
from time import time

def request_pages(df_urls):
    web_pages = []
    for i in range(len(df_urls)):
        u = df_urls.iloc[i]
        url = u[2]
        short_url = url[0:50]
        short_url = short_url.replace("https//", "")
        short_url = short_url.replace("http//", "")
        n = 0
        # Allow for a maximum of 5 download failures
        stop_sec=3 # Initial max wait time in seconds
        while n<3:
            try:
                r = requests.get(url, timeout=(stop_sec))
                if r.status_code == 408:
                    print("-->HTML ERROR 408", short_url)
                    raise ValueError()
                if r.status_code == 200:
                    print("Obtained: "+short_url)
                else:
                    print("-->Web page: "+short_url+" status code:", \
                          r.status_code)
                n=99
                continue # Skip this page
            except:
                n += 1
                # Timeout waiting for download
                t0 = time()
                tlapse = 0
                print("Waiting", stop_sec, "sec")
                while tlapse<stop_sec:
                    tlapse = time()-t0
        if n != 99:
            # download failed skip this page
           continue
        #   Page obtained successfully
        html_page = r.text
        page_text = clean_html(html_page)
        web_pages.append([url, page_text])
    df_www = pd.DataFrame(web_pages, columns=['url', 'text'])
    n_total = len(df_urls)
    # Remove duplicates
    df_www = df_www.drop_duplicates('url')
    n_unique = len(df_urls)
    print("Found a total of", n_total, " web pages, of which", n_unique,\
          " were unique.")
    return df_www
def newsapi_get_urls(search_words, agency_urls):
    if len(search_words)==0 or agency_urls==None:
        return None
    print("Searching agencies for pages containing:", search_words)
    # This is my API key, each user must request their own
    # API key from https://newsapi.org/account
    api = NewsApiClient(api_key='6f174feb5d05447d920d538d45718afa')
    api_urls = []
    # Iterate over agencies and search words to pull more url's
    # Limited to 1,000 requests/day - Likely to be exceeded
    for agency in agency_urls:
        domain = agency_urls[agency].replace("http://", "")
        print(agency, domain)
        for word in search_words:
            # Get articles with q= in them, Limits to 20 URLs
            try:
                articles = api.get_everything(q=word, language='en',\
                                              sources=agency, domains=domain)
            except:
                print("--->Unable to pull news from:", agency, "for", word)
                continue
            # Pull the URL from these articles (limited to 20)
            d = articles['articles']
            for i in range(len(d)):
                url = d[i]['url']
                api_urls.append([agency, word, url])
    df_urls = pd.DataFrame(api_urls, columns=['agency', 'word', 'url'])
    n_total = len(df_urls)
    # Remove duplicates
    df_urls = df_urls.drop_duplicates('url')
    n_unique = len(df_urls)
    print("\nFound a total of", n_total, " URLs, of which", n_unique,\
          " were unique.")
    return df_urls
def clean_html(html):
    # First we remove inline JavaScript/CSS:
    pg = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    pg = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", pg)
    # Next we can remove the remaining tags:
    pg = re.sub(r"(?s)<.*?>", " ", pg)
    # Finally, we deal with whitespace
    pg = re.sub(r"&nbsp;", " ", pg)
    pg = re.sub(r"&rsquo;", "'", pg)
    pg = re.sub(r"&ldquo;", '"', pg)
    pg = re.sub(r"&rdquo;", '"', pg)
    pg = re.sub(r"\n", " ", pg)
    pg = re.sub(r"\t", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    return pg.strip()

def my_analyzer(s):
    # Synonym List

    # I added some synonym words like "speed up" to "accelerate", "injured" to "hurt"...
    syns = {'veh': 'vehicle', 'car': 'vehicle', 'chev': 'cheverolet', \
            'chevy': 'cheverolet', 'air bag': 'airbag', \
            'seat belt': 'seatbelt', "n't": 'not', 'to30': 'to 30', \
            'wont': 'would not', 'cant': 'can not', 'cannot': 'can not', \
            'couldnt': 'could not', 'shouldnt': 'should not', \
            'wouldnt': 'would not', 'straightforward': 'straight forward', \
            'mileage': 'mile', 'injured': 'hurt','speed up':'accelerate','fixed':'repaired','skid':'brake',}

    # Preprocess String s
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    # Tokenize
    tokens = word_tokenize(s)
    # tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and \
              ("''" != word) and ("``" != word) and \
              (word != 'description') and (word != 'dtype') \
              and (word != 'object') and (word != "'s")]

    # Map synonyms
    for i in range(len(tokens)):
        if tokens[i] in syns:
            tokens[i] = syns[tokens[i]]

    # Remove stop words
    punctuation = list(string.punctuation) + ['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    others = ["'d", "co", "ed", "put", "say", "get", "can", "become", \
              "los", "sta", "la", "use", "iii", "else"]
    stop = stopwords.words('english') + punctuation + pronouns + others
    filtered_terms = [word for word in tokens if (word not in stop) and \
                      (len(word) > 1) and (not word.replace('.', '', 1).isnumeric()) \
                      and (not word.replace("'", '', 2).isnumeric())]

    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N': wn.NOUN, 'J': wn.ADJ, 'V': wn.VERB, 'R': wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos = tagged_token[1]
        pos = pos[0]
        try:
            pos = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    return stemmed_tokens


def display_topics(lda, terms, n_terms=15):
    for topic_idx, topic in enumerate(lda):
        if topic_idx > 8:
            break
        message = "Topic #%d: " % (topic_idx + 1)
        print(message)
        abs_topic = abs(topic)
        topic_terms_sorted = \
            [[terms[i], topic[i]] \
             for i in abs_topic.argsort()[:-n_terms - 1:-1]]
        k = 5
        n = int(n_terms / k)
        m = n_terms - k * n
        for j in range(n):
            l = k * j
            message = ''
            for i in range(k):
                if topic_terms_sorted[i + l][1] > 0:
                    word = "+" + topic_terms_sorted[i + l][0]
                else:
                    word = "-" + topic_terms_sorted[i + l][0]
                message += '{:<15s}'.format(word)
            print(message)
        if m > 0:
            l = k * n
            message = ''
            for i in range(m):
                if topic_terms_sorted[i + l][1] > 0:
                    word = "+" + topic_terms_sorted[i + l][0]
                else:
                    word = "-" + topic_terms_sorted[i + l][0]
                message += '{:<15s}'.format(word)
            print(message)
        print("")
    return

def term_dic(tf, terms, scores=None):
    td = {}
    for i in range(tf.shape[0]):
        # Iterate over the terms with nonzero scores
        # print(type(tf),type(tf[i]))
        # print(tf,'\n\n\n')
        # print(tf[i])
        term_list = tf[i].nonzero()[1]
        if len(term_list)>0:
            if scores==None:
                for t in np.nditer(term_list):
                    if td.get(terms[t]) == None:
                        td[terms[t]] = tf[i,t]
                    else:
                        td[terms[t]] += tf[i,t]
            else:
                for t in np.nditer(term_list):
                    score = scores.get(terms[t])
                    if score != None:
                        # Found Sentiment Word
                        score_weight = abs(scores[terms[t]])
                        if td.get(terms[t]) == None:
                            td[terms[t]] = tf[i,t] * score_weight
                        else:
                            td[terms[t]] += tf[i, t] * score_weight
    return td
def my_preprocessor(s):
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    return (s)


df=pd.read_excel('HondaComplaints.xlsx')
df_sentiment=pd.read_excel('afinn_sentiment_words.xlsx')
attribute_map={
    'description':[3,(''),[0,0]],
    'Make':[1,('HONDA','ACURA'),[0,0]],
    'Model':[2,('TL','ODYSSEY','CR-V','CL','CIVIC','ACCORD'),[0,0]],
    'Year':[2,(2001,2002,2003),[0,0]],
    'abs':[1,('Y','N'),[0,0]],
    'cruise':[1,('Y','N'),[0,0]],
    'crash':[1,('Y','N'),[0,0]],
    'mph':[0,(0,80),[0,0]],
    'mileage':[0,(0,200000),[0,0]],
    'T1':[0,(-1e+8,1e+8),[0,0]],
    'T2':[0,(-1e+8,1e+8),[0,0]],
    'T3':[0,(-1e+8,1e+8),[0,0]],
    'T4':[0,(-1e+8,1e+8),[0,0]],
    'T5':[0,(-1e+8,1e+8),[0,0]],
    'T6':[0,(-1e+8,1e+8),[0,0]],
    'T7':[0,(-1e+8,1e+8),[0,0]]
}


description=df['description']
m_features = None
comments = df['description']
n_topics =  7
n_comments  = len(df['description'])


cv = CountVectorizer(max_df=0.95, min_df=2, max_features=m_features,\
                     analyzer=my_analyzer, ngram_range=(1,2))
tf    = cv.fit_transform(comments)
terms = cv.get_feature_names()
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))
print("")

print("Conducting Term/Frequency Matrix using TF-IDF")
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
tf         = tfidf_vect.fit_transform(tf)

term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",\
            tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[i][0], \
          term_idf_scores[i][1]))

uv = TruncatedSVD(n_components=7, algorithm='arpack',\
                            tol=0, random_state=12345)
U = uv.fit_transform(tf)

# Display the topic selections
print("\n********** GENERATED TOPICS **********")
display_topics(uv.components_, terms, n_terms=15)

topics = [0] * n_comments
topic_counts = [0] * (n_topics + 1)
for i in range(n_comments):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
    topic_counts[topics[i]] += 1

print('{:<6s}{:>8s}{:>8s}'.format("TOPIC", "COMMENTS", "PERCENT"))
for i in range(n_topics):
    print('{:>3d}{:>10d}{:>8.1%}'.format((i + 1), topic_counts[i], \
                                         topic_counts[i] / n_comments))

# Create comment_scores[] and assign the topic groups
comment_scores = []
for i in range(n_comments):
    u = [0] * (n_topics + 1)
    u[0] = topics[i]
    for j in range(n_topics):
        u[j + 1] = U[i][j]
    comment_scores.append(u)

# Augment Dataframe with topic group information
cols = ["topic"]
for i in range(n_topics):
    s = "T" + str(i + 1)
    cols.append(s)
df_topics = pd.DataFrame.from_records(comment_scores, columns=cols)
df = df.join(df_topics)


# start sentiment analysis
sentiment_dic={}
for i in range(len(df_sentiment)):
    sentiment_dic[df_sentiment.iloc[i][0]]=df_sentiment.iloc[i][1]
cv_sen = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     preprocessor=my_preprocessor,ngram_range=(1,2))
tf_sen=cv_sen.fit_transform(df['description'])

s_terms=cv_sen.get_feature_names()
n_description=tf_sen.shape[0]
n_terms_sen=tf_sen.shape[1]
sentiment_score = [0]*n_description
min_list, max_list = [],[]
avg_sentiment, min, max = 0,0,0
for i in range(n_description):
    n_sw=0
    term_list = tf_sen[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            score=sentiment_dic.get(s_terms[t])
            if score!= None:
                sentiment_score[i]+=score *tf_sen[i,t]
                n_sw+=tf_sen[i,t]
    if n_sw>0:
        sentiment_score[i]=sentiment_score[i]/n_sw
df_senscore=pd.DataFrame(sentiment_score,columns=['sentiment score'])
df=df.join(df_senscore)


# classify topic based on the probability

df['topic']=0
for ix, row in df.iterrows():
    mx=row[['T1','T2','T3','T4','T5','T6','T7']].max()
    b=(row==mx).idxmax(axis=1)
    df.loc[ix, 'topic'] = b
# save the data output of NLP
df.to_csv('after_NLP_data.csv',index=False)

# scale data
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                           interval_scale=None, drop=False, display=True)

df_tree = rie.fit_transform(df)
y= df_tree['crash']
X = df_tree.drop('crash',axis=1)

# find the best tree depth
depth_list = [3, 5, 6, 7, 8, 10, 12, 15, 20, 25]
score_list = ['accuracy', 'recall', 'precision', 'f1']
for d in depth_list:
    print("\nMaximum Tree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, \
                                 min_samples_split=5,random_state=12345)
    dtc = dtc.fit(X, y)
    scores = cross_validate(dtc, X, y, scoring=score_list, \
                            return_train_score=False, cv=10)

    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_" + s
        mean = scores[var].mean()
        std = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

# split and validate
X_train_tree, X_validate_tree, y_train_tree, y_validate_tree = \
             train_test_split(X, y,test_size = 0.3, random_state=7)
dtc = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,min_samples_split=5)
dtc = dtc.fit(X_train_tree, y_train_tree)
DecisionTree.display_binary_split_metrics(dtc, X_train_tree, y_train_tree, X_validate_tree,  y_validate_tree)





# News Agencies used by API
agency_urls = {
'huffington': 'http://huffingtonpost.com',
'reuters': 'http://www.reuters.com',
'cbs-news': 'http://www.cbsnews.com',
'usa-today': 'http://usatoday.com',
'cnn': 'http://cnn.com',
'npr': 'http://www.npr.org',
'wsj': 'http://wsj.com',
'fox': 'http://www.foxnews.com',
'abc': 'http://abc.com',
'abc-news': 'http://abcnews.com',
'abcgonews': 'http://abcnews.go.com',
'nyt': 'http://nytimes.com',
'washington-post': 'http://washingtonpost.com',
'us-news': 'http://www.usnews.com',
'msn': 'http://msn.com',
'pbs': 'http://www.pbs.org',
'nbc-news': 'http://www.nbcnews.com',
'enquirer': 'http://www.nationalenquirer.com',
'la-times': 'http://www.latimes.com'
}




search_words = ['takata']
df_urls = newsapi_get_urls(search_words, agency_urls)
print("Total Articles:", df_urls.shape[0])

print("Agency:", df_urls.iloc[0]['agency'])
print("Search Word:", df_urls.iloc[0]['word'])
print("URL:", df_urls.iloc[0]['url'])

# Download Discovered Pages
df_www = request_pages(df_urls)
# Store in Excel File
df_www.to_excel('df_www.xlsx')

for i in range(df_www.shape[0]):
    short_url = df_www.iloc[i]['url']
    short_url = short_url.replace("https://", "")
    short_url = short_url.replace("http://", "")
    short_url = short_url[0:60]
    page_char = len(df_www.iloc[i]['text'])
    print("{:<60s}{:>10d} Characters".format(short_url, page_char))
