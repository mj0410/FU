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
import configargparse
from utils import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from models_sherlock import *

from extract.feature_extraction.topic_features_LDA import *
from extract.feature_extraction.sherlock_features import *

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
    print("df.shape[1]", n)
    MAX_COL_COUNT = len(df.columns)

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
    return feature_dic, np.zeros(MAX_COL_COUNT), torch.tensor([1]*n + [0]*(MAX_COL_COUNT-n), dtype=torch.uint8)

def evaluate(df, classifier):

    feature_dic, labels, mask = extract(df)

    #emissions = classifier(feature_dic).view(1, MAX_COL_COUNT, -1)
    #mask = mask.view(1, MAX_COL_COUNT)
    #pred = model.decode(emissions, mask)[0]
    pred = []
    pred_tensor = classifier(feature_dic)
    # print("pred_tensor type", type(pred_tensor))
    pred.extend(pred_tensor.detach().numpy())
    pred = np.argmax(pred, axis=1)

    return label_enc.inverse_transform(pred)


if __name__ == "__main__":

    #################### 
    # Load configs
    #################### 
    p = configargparse.ArgParser()
    p.add('-c', '--config_file', required=True, is_config_file=True, help='config file path')
    p.add('-i', '--input_file_path', type=str_or_none, default=None, help='path to input tables')
    p.add('-dt', '--deeptable_result', type=str_or_none, default=None, help='prediction results from DeepTable')

    # general configs
    p.add('--TYPENAME', type=str, help='Name of valid types', env_var='TYPENAME')

    # sherlock configs
    p.add('--sherlock_feature_groups', nargs='+', default=['char','rest','par','word'])
    p.add('--topic', type=str_or_none, default=None)

    # exp configs
    p.add('--model', type=str, help='load pretrained model')

    args = p.parse_args()

    inp_path = args.input_file_path
    dt_pred = args.deeptable_result
    TYPENAME = args.TYPENAME

    sherlock_feature_groups = args.sherlock_feature_groups
    topic_name = args.topic
    model_name = args.model

    config_name = os.path.split(args.config_file)[-1].split('.')[0]

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

    classifier = build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim,
                                dropout_ratio=0.2).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)

    sherlock_model_loc = join(os.environ['BASEPATH'], 'model', 'pre_trained_sherlock', TYPENAME)
    classifier.load_state_dict(torch.load(join(sherlock_model_loc, '{}.pt'.format(model_name)), map_location=device))
    classifier.eval()

    table_name, pred = [], []

    if dt_pred is not None:
        dt_result = pd.read_csv(join(os.environ['ORIGIN'], '{}.csv'.format(dt_pred)))
        print('number of tables : ', len(dt_result))

        for index, row in dt_result.iterrows():
            print(row[1])
            df = pd.read_csv(join(row[0], row[1]))
            if row[2]==1:
                df = df.T.reset_index()
            prediction = evaluate(df, classifier)
            pred.append(prediction)
            table_name.append(row[1])

    if inp_path is not None:
        table_list = listdir(inp_path)
        print('number of tables : ', len(table_list))

        for table in table_list:
            print(table)
            df = pd.read_csv(join(inp_path, table))
            prediction = evaluate(df, classifier)
            pred.append(prediction)
            table_name.append(table)

    result = pd.DataFrame({'table' : table_name, 'prediction' : pred})
    result_path = join(os.environ['ORIGIN'], 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result.to_csv(join(result_path, 'sherlock_prediction.csv'))
    t2 = time.time()
    print("prediction is done in {} sec".format(int(t2 - t1)))


