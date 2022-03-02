from time import time
import os, sys
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)
from os.path import join
import numpy as np
from os import listdir
import json, pickle
#import copy
#import datetime
import argparse
from utils import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random

from models_sherlock import *
from torchcrf import CRF

from extract.feature_extraction.topic_features_LDA import *
from extract.feature_extraction.sherlock_features import *
from data_type_detection import *

# =============
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import ConcatDataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============

def extract(df):

    df_dic = {'df':df, 'locator':'None', 'dataset_id':'None'}
    feature_dic = {}
    n = df.shape[1]
    #print("df.shape[1]", n)
    MAX_COL_COUNT = len(df.columns) #

    # topic vectors
    if topic_name:
        topic_features = extract_topic_features_pred(df_dic)
        topic_vec = pad_vec(topic_features.loc[0,'table_topic'])
        feature_dic['topic'] = torch.FloatTensor(np.vstack((np.tile(topic_vec,(n,1)), np.zeros((MAX_COL_COUNT - n, topic_dim)))))

    # sherlock vectors
    sherlock_features = extract_sherlock_features_pred(df_dic)
    for f_g in feature_group_cols:
        temp = sherlock_features[feature_group_cols[f_g]].to_numpy()
        temp = np.vstack((temp, np.zeros((MAX_COL_COUNT - n, temp.shape[1])))).astype('float')
        temp = np.nan_to_num(temp)
        feature_dic[f_g] = torch.FloatTensor(temp)

    # dictionary of features, labels, masks
    return MAX_COL_COUNT, feature_dic, np.zeros(MAX_COL_COUNT), torch.tensor([1]*n + [0]*(MAX_COL_COUNT-n), dtype=torch.uint8)

def evaluate(df, classifier, model):

    MAX_COL_COUNT, feature_dic, labels, mask = extract(df)

    emissions = classifier(feature_dic).view(1, MAX_COL_COUNT, -1)
    mask = mask.view(1, MAX_COL_COUNT)
    pred = model.decode(emissions, mask)[0]

    return label_enc.inverse_transform(pred)

def count_missing_values(df):
    col_num, cols_with_missing = [], []
    for i in range(len(df.columns)):
        c = df.columns[i]
        p = round(df[df[c] == ''].shape[0]/len(df), 2)
        if p>=0.8:
            col_num.append(i)
    if len(col_num)==0:
        col_num = ''
    return col_num

def part_prediction(df, num_row, classifier, model):
    pred_part = []
    print("number of rows : ", len(df), '\n')
    idx_list = list(df.index)
    random.shuffle(idx_list)

    chunks = [idx_list[x:x + num_row] for x in range(0, len(idx_list), num_row)]
    #if len(chunks)>5 :
    #    chunks = chunks[:5]
    n = 1
    print(len(chunks))
    for idx in chunks:
        print("{}th running".format(n))
        n = n + 1
        df_part = df.loc[idx]
        prediction_part = evaluate(df_part, classifier, model)
        pred_part.append(prediction_part)

    #print("prediction per part of table \n")
    #print(pred_part)

    pred_part_arr = np.array(pred_part)
    prediction = []
    for col in pred_part_arr.T:
        elements, counts = np.unique(col, return_counts=True)
        prediction.append(elements[np.argmax(counts)])

    return prediction


if __name__ == "__main__":

    #################### 
    # Load configs
    #################### 
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_file_path', type=str_or_none, default=None, help='path to input tables')
    p.add_argument('-n', '--num_rows', type=int, default=None, help='number of rows if input tables are large')
    p.add_argument('-dt', '--deeptable_result', type=str_or_none, default=None, help='prediction results from DeepTable')
    p.add_argument('-o', '--output', type=str, help='output file name')

    # sherlock configs
    p.add_argument('--sherlock_feature_groups', nargs='+', default=['char','rest','par','word'])
    p.add_argument('-t', '--topic', type=str_or_none, default=None)

    # exp configs
    p.add_argument('-m', '--model', type=str, help='load pretrained model')

    args = p.parse_args()

    inp_path = args.input_file_path
    num_row = args.num_rows
    dt_pred = args.deeptable_result
    TYPENAME = os.environ['TYPENAME']

    sherlock_feature_groups = args.sherlock_feature_groups
    topic_name = args.topic
    model_name = args.model
    filename = args.output

    print('\n--------------------------------------')
    print("DeepTable prediction : {}.csv".format(dt_pred))
    print("TYPENAME : {}".format(TYPENAME))
    print("pre trained model : {}".format(model_name))
    print('--------------------------------------\n')

    #################### 
    # Preparations
    #################### 
    valid_types = get_valid_types(TYPENAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PyTorch device={}".format(device))

    if topic_name:
        model_loc = join(os.environ['BASEPATH'], 'topic_model', 'LDA_cache', TYPENAME, topic_name)
        kwargs_file = join(model_loc, "{}.pkl".format(topic_name))
        with open(kwargs_file, 'rb') as f:
            kwargs = pickle.load(f)
        topic_dim = kwargs['tn']
        print("topic_name : ", topic_name)
        print("topic_dim : ", topic_dim)

        pad_vec = lambda x: np.pad(x, (0, topic_dim - len(x)),
                                   'constant',
                                   constant_values=(0.0, 1 / topic_dim))
    else:
        topic_dim = None

    feature_group_cols = {}
    for f_g in sherlock_feature_groups:
        feature_group_cols[f_g] = list(pd.read_csv(join(os.environ['BASEPATH'],
                                                        'configs', 'feature_groups',
                                                        "{}_col.tsv".format(f_g)),
                                                   sep='\t', header=None,
                                                   index_col=0)[1])


    ##################################
    ########### Load data ############
    ##################################

    time_record = {}

    # 1. Dataset
    t1 = time.time()
    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    ######################
    ##### Load model #####
    ######################

    pre_trained_loc = join(BASEPATH, 'model', 'pre_trained_CRF', TYPENAME)

    classifier = build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim,
                                dropout_ratio=0.2).to(device)
    model = CRF(len(valid_types), batch_first=True).to(device)

    loaded_params = torch.load(join(pre_trained_loc, '{}.pt'.format(model_name)), map_location=device)
    classifier.load_state_dict(loaded_params['col_classifier'])
    model.load_state_dict(loaded_params['CRF_model'])

    classifier.eval()
    model.eval()

    table_path, table_name, dt_label, pred, colidx_with_missing, dtype = [], [], [], [], [], []

    if dt_pred is not None:
        dt_result = pd.read_csv(join(os.environ['ORIGIN'], '{}.csv'.format(dt_pred)))
        print('number of tables : ', len(dt_result), '\n')
        ans = input(
            "If input table is large (>{} rows), do you want to split the table and get average result for better prediction?".format(num_row))

        for index, row in dt_result.iterrows():
            print(row[1])
            if row[2] == 1:
                df = pd.read_csv(join(row[0], row[1]), index_col=0, keep_default_na=False)
                df = df.T.reset_index(drop=True)
                print("[Transposed] num of cols : {}".format(len(df.columns)))
                #print("transposed : \n")
                #print(df.head(3))
            else:
                df = pd.read_csv(join(row[0], row[1]), keep_default_na=False)
                print("[Not transposed] num of cols : {}".format(len(df.columns)))
                #print("Not transposed : \n")
                #print(df.head(3))

            if num_row is not None and len(df) > num_row and ans == 'y' :
                #ans = input("The table {} is large (>{} rows). Do you want to split the table and get average result for better prediction? ".format(row[1], num_row))
                print("split the table into {} rows".format(num_row))
                prediction = part_prediction(df, num_row, classifier, model)

                #else:
                 #   print("continue without split")
                  #  prediction = evaluate(df, classifier, model)

            else:
                prediction = evaluate(df, classifier, model)
                prediction = str_preprocess_pred(str(prediction))

            colidx_with_missing.append(count_missing_values(df))

            pred.append(prediction)
            table_name.append(row[1])
            table_path.append(row[0])
            dt_label.append(row[2])
            dtype.append(data_type(df))
            #print(row[1], prediction, '\n')

        result = pd.DataFrame({'dir': table_path, 'table': table_name, 'table_orientation' : dt_label,
                               'semantic_type': pred, 'colidx_with_missing': colidx_with_missing, 'data_type':dtype})

    elif inp_path is not None:
        table_list = listdir(inp_path)
        print('number of tables : ', len(table_list))
        ans = input(
            "If input table is large (>{} rows), do you want to split the table and get average result for better prediction?".format(
                num_row))

        for table in table_list:
            print(table)
            df = pd.read_csv(join(inp_path, table), keep_default_na=False)

            if num_row is not None and len(df) > num_row and ans == 'y' :
                #ans = input("The table {} is large (>{} rows). Do you want to split the table and get average result for better prediction? ".format(row[1], num_row))
                print("split the table into {} rows".format(num_row))
                prediction = part_prediction(df, num_row, classifier, model)

                #else:
                #    #print("continue without split")
                #    prediction = evaluate(df, classifier, model)

            else:
                prediction = evaluate(df, classifier, model)
                prediction = str_preprocess_pred(str(prediction))

            pred.append(prediction)
            table_name.append(table)
            colidx_with_missing.append(count_missing_values(df))
            table_path.append(inp_path)
            dtype.append(data_type(df))

        result = pd.DataFrame({'dir': table_path, 'table': table_name, 'semantic_type': pred,
                               'colidx_with_missing': colidx_with_missing, 'data_type':dtype})

    result_path = join(os.environ['ORIGIN'], 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #result['semantic_type'] = result['semantic_type'].map(str_preprocess)
    result.to_csv(join(result_path, 'prediction_{}_{}.csv'.format(TYPENAME, filename)), index=False)
    t2 = time.time()
    print("prediction is done in {} sec".format(int(t2 - t1)))



