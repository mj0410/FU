import os
import sys
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)

from os.path import join
from os import listdir
import argparse
import pandas as pd
import pickle
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import math
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from utils import dic2name, int_or_none
import time

TYPENAME = os.environ['TYPENAME']
LDANAME = os.environ['LDANAME']
LDA_CACHE = join('LDA_cache', TYPENAME, LDANAME)

#nltk.download('stopwords')
#nltk.download('punkt')

def clean(s):
    tokens = nltk.word_tokenize(s.lower())
    tokens_clean = [token for token in tokens if token not in stopwords.words('english')]
    tokens_stemmed = [PorterStemmer().stem(token) for token in tokens_clean]
    return tokens_stemmed


def tokenize(col, **kwargs):
    threshold = int(kwargs['thr'])
    ret = []
    for st in col:
        if len(st)> threshold:
            # tokenize the string if longer than threshold
            # and append a longstr tag
            ret.extend(clean(st))
            if threshold > 0:
                ret.append('longstr')
        else:
            ret.append(st.lower())
    return ret


def process_col(col, **kwargs):

    numeric = kwargs['num']
    # process the cols to return a bags of word representation
    if col.dtype == 'int64' or col.dtype =='float64':
        if numeric == 'directstr':
            return list(col.astype(str))
        elif numeric == 'placeholder':
            return [str(col.dtype)] * len(col)
        
    if col.dtype == 'object':
        return tokenize(list(col.astype(str)), **kwargs)
    
    else:
        return list(col.astype(str))
       
    return col

def corpus_iter(table_df, batch_size, **kwargs):
    # iterate through tables in a directory, collect all values in each table.
    f_list = []
    for index, row in table_df.iterrows():
        file = join(row['locator'], row['dataset_id'])
        rows = eval(row['row_index'])
        temp_list = [file, rows]
        f_list.append(temp_list)

    l = len(f_list)
    print("total # of tables", l)

    for b in range(math.ceil(l/float(batch_size))):    

        corpus = []
        for f in f_list[b*batch_size: min((b+1)*batch_size, l)]:
            #try
            df = pd.read_csv(f[0], keep_default_na=False) #, index_col=0, error_bad_lines=False)
            df = df.iloc[f[1]]
            #except Exception as e:
             #   print("Exception", e)
              #  continue
                
            table_seq = []
            for col in df.columns:
                processed_col = process_col(df[col], **kwargs)
                table_seq.extend(processed_col)
            #print("table_sep : {}".format(table_seq))
            corpus.append(table_seq)

        yield corpus

def train_LDA(model_name, table_df, batch_size, use_dictionary=False, **kwargs):

    model_name = model_name
    print("Model: ", model_name)
    topic_num = kwargs['tn']

    # Pass 1 get the dictionary
    if use_dictionary=='True':
        dic = Dictionary.load(join(LDA_CACHE, 'dictionary_{}'.format(model_name)))
    else:

        dic = Dictionary([])
        b = 0
        for corpus in corpus_iter(table_df, batch_size, **kwargs):
            # print("first corpus : ", corpus)
            dic.add_documents(corpus)
            print('Dictionary batch {}: current dic size {}'.format(b, len(dic)))
            b+=1
            
        # save dictionary
        dic.save(join(LDA_CACHE, 'dictionary_{}'.format(model_name)))

    print("Dictionary size", len(dic))
    
    # Pass 2 train LDA
    whole_corpus = corpus_iter(table_df, batch_size, **kwargs)
    # print("second corpus : ", whole_corpus)
    first_batch = next(whole_corpus)
    first_bow = [dic.doc2bow(text, allow_update=False) for text in first_batch]
    #print(first_bow)

    lda = LdaModel(first_bow, id2word=dic, num_topics=topic_num, minimum_probability=0.0)
    batch_no = 0
    print('LDA update batch {}'.format(batch_no))

    for batch in whole_corpus:
        batch_bow = [dic.doc2bow(text, allow_update=False) for text in batch]
        #print(corpus_bow)
        lda.update(batch_bow)
        batch_no +=1
        print('LDA update batch {}'.format(batch_no))
    
    # Save model to disk.
    temp_file = join(LDA_CACHE, model_name)
    lda.save(temp_file)
    
    print("Training done. Batch_size: {}, long str tokenization threshold: {}, numerical representations: {}.\
          \nTotal size of dictionary: {}".format(batch_size, kwargs['thr'], kwargs['num'], len(dic)))
    return


if __name__ == "__main__":

    if not os.path.exists(LDA_CACHE):
        os.makedirs(LDA_CACHE)

    # Get corpus
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', type=str, default='LDA_model')
    parser.add_argument('-cv', '--cross_validation', type=int_or_none, default=None)
    parser.add_argument('-s', '--train_test_split_file', type=str, help='train / test split file name')
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=5000)
    parser.add_argument('-l', '--long_threshold', nargs='?', type=int, default=0, help='Long strings will be tokenized and tag as longstr')
    parser.add_argument('-num', '--numeric_rep', nargs='?', type=str, default='directstr', choices=['directstr', 'placeholder', 'bin'],
                         help='Convert numbers to strings directly or replace with int64/float64')
    
    parser.add_argument('--topic_num', nargs='?', type=int, default=100, help='Number of topics')

    parser.add_argument( '--use_dictionary', nargs='?', type=str, default='False', choices=['True', 'False'],
                         help='if set to True, use existing dictionary in LDA_cache')
    args = parser.parse_args()

    split_file = args.train_test_split_file
    cv = args.cross_validation

    if cv is None:
        split_dir = join(BASEPATH, 'extract/out/train_test_split', '{}_{}.csv'.format(split_file, TYPENAME))
        split = pd.read_csv(split_dir)
        train = split[split['split'] == 'train']
    else:
        split_dir = join(BASEPATH, 'extract/out/train_test_split', '{}_{}fold_{}.csv'.format(split_file, cv, TYPENAME))
        split = pd.read_csv(split_dir)
        train = split[split['cv'] != 'test']

    print("number of table", len(train))
    
    batch_size = args.batch_size
    model_name = args.model_name

    start_time = time.time()

    kwargs = {'thr': args.long_threshold, 'num': args.numeric_rep, 'tn':args.topic_num}

    kwargs_file = join(LDA_CACHE, "{}.pkl".format(model_name))
    with open(kwargs_file, 'wb') as f:
        pickle.dump(kwargs, f)

    train_LDA(model_name, train, batch_size, args.use_dictionary, **kwargs)

    end_time = time.time()
    print("Feature extraction done ({} sec)".format(int(end_time - start_time)))

