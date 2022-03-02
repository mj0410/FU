from time import time
import os, sys
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)
from os.path import join
import numpy as np
import json, pickle
import copy
import datetime
import configargparse
from statistics import mean
import random
from utils import str2bool, str_or_none, int_or_none, name2dic, get_valid_types
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter

import datasets
from models_sherlock import FeatureEncoder, SherlockClassifier, build_sherlock
from sklearn.metrics import classification_report

# =============
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============

if __name__ == "__main__":


    #################### 
    # Load configs
    #################### 
    p = configargparse.ArgParser()
    p.add('-c', '--config_file', required=True, is_config_file=True, help='config file path')

    # general configs
    p.add('--n_worker', type=int, default=2, help='# of workers for dataloader')
    p.add('--TYPENAME', type=str, help='Name of valid types', env_var='TYPENAME')
    p.add('--train_test_split', type=str, help='Name of output from split_train_test.py')
    p.add('--header', type=str, help='Name of output from extract_header.py')
    p.add('--sherlock_feature', type=str, help='extracted sherlock feature file name')
    p.add('--topic_feature', type=str, help='extracted topic feature file name')

    # NN configs
    p.add('--epochs', type=int, default=100)
    p.add('--learning_rate', type=float, default=1e-4)
    p.add('--decay', type=float, default=1e-4)
    p.add('--dropout_rate', type=float, default=0.35)
    p.add('--batch_size', type=int, default=256, help='# of col in a batch')
    p.add('--patience', type=int, default=100, help='patience for early stopping')

    # sherlock configs
    p.add('--sherlock_feature_groups', nargs='+', default=['char','rest','par','word'])
    p.add('--topic', type=str_or_none, default=None)

    # exp configs
    p.add('--cross_validation', type=int_or_none, default=None)
    p.add('--num', type=int, default=100, help='number of experiments')
    p.add('--model_name', type=str, default='sherlock', help='saved model name')


    args = p.parse_args()


    n_worker = args.n_worker
    TYPENAME = args.TYPENAME
    train_test_file = args.train_test_split
    header_file = args.header
    sherlock_feature_file = args.sherlock_feature
    topic_feature_file = args.topic_feature

    ## Loading Hyper parameters
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.decay
    dropout_ratio = args.dropout_rate    
    batch_size = args.batch_size
    patience = args.patience
    cross_validation = args.cross_validation
    num_exp = args.num

    sherlock_feature_groups = args.sherlock_feature_groups
    topic_name = args.topic

    config_name = os.path.split(args.config_file)[-1].split('.')[0]

    #################### 
    # Preparations
    #################### 
    valid_types = get_valid_types(TYPENAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PyTorch device={}".format(device))

    if topic_name:
        model_loc = os.environ['TOPICMODELPATH']
        LDANAME = os.environ['LDANAME']
        kwargs_file = join(model_loc, LDANAME, "{}.pkl".format(topic_name))
        with open(kwargs_file, 'rb') as f:
            kwargs = pickle.load(f)
        topic_dim = kwargs['tn']
        print("topic_name : ", topic_name)
        print("topic_dim : ", topic_dim)
    else:
        topic_dim = None


    # tensorboard logger
    currentDT = datetime.datetime.now()
    DTString = '-'.join([str(x) for x in currentDT.timetuple()[:5]])
    logging_base = 'sherlock_log' #if device == torch.device('cpu') else 'sherlock_cuda_log'

    logging_name = args.model_name

    logging_path = join(os.environ['BASEPATH'],'results', logging_base, TYPENAME, logging_name)

    print('\nlogging_name', logging_name)

    time_record = {}

    # 1. Dataset

    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    train_test_path = join(os.environ['BASEPATH'], 'extract', 'out', 'train_test_split')

    split = pd.read_csv(join(train_test_path, '{}.csv'.format(train_test_file)))


    tableFeatures = datasets.TableFeatures(header_file,
                                           sherlock_feature_file,
                                           sherlock_feature_groups,
                                           extracted_feature_topic=topic_feature_file,
                                           topic_feature=topic_name,
                                           label_enc=label_enc,
                                           id_filter=None,
                                           max_col_count=None)

    # 2. Models
    result_list = []
    best_acc = None
    print("run {} experiments".format(num_exp))

    for k in range(num_exp):

        #val_result = {'cv': [], 'acc': [], 'best_acc_epoch':[], 'best_loss_epoch':[]}
        print('Conducting cross validation, current {}th in {} experiments'.format(k+1, num_exp))

        print("Creating Dataset object...")
        Dataset_start_time = time()

        #random_int_list = random.sample(range(1, 6), 1)
        random_int_list = random.sample(range(1, cross_validation+1), round(cross_validation*0.2))
        random_str_list = list(map(str, random_int_list))

        val = split[split['cv'].isin(random_str_list)]
        test = split[split['cv'] == 'test']

        random_str_list.append('test')
        train = split[~split['cv'].isin(random_str_list)]

        print('data length:')
        print(len(train), len(val), len(test))

        train_list, val_list, test_list = [], [], []

        train_tf = copy.copy(tableFeatures).set_filter(train['dataset_id'], train['row_index']).to_col()
        train_list.append(train_tf)

        train_end_time = time()
        print("Train datasets are created. ({} sec)".format(int(train_end_time - Dataset_start_time)))

        val_tf = copy.copy(tableFeatures).set_filter(val['dataset_id'], val['row_index']).to_col()
        val_list.append(val_tf)

        val_end_time = time()
        print("Val datasets are created. ({} sec)".format(int(val_end_time - train_end_time)))

        test_tf = copy.copy(tableFeatures).set_filter(test['dataset_id'], test['row_index']).to_col()
        test_list.append(test_tf)

        test_end_time = time()
        print("Test datasets are created. ({} sec)".format(int(test_end_time - val_end_time)))

        #print("Dataset length:")
        #print(len(train_list), len(val_list), len(test_list))

        train_dataset = ConcatDataset(train_list)
        val_dataset = ConcatDataset(val_list)
        test_dataset = ConcatDataset(test_list)

        Dataset_end_time = time()
        print("Dataset objects are created. ({} sec)".format(int(Dataset_end_time - Dataset_start_time)))

        print("model training...")

        classifier = build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim,
                                    dropout_ratio=dropout_ratio).to(device)
        loss_func = nn.CrossEntropyLoss().to(device)

        if not os.path.exists(join(logging_path, "writer")):
            os.makedirs(join(logging_path, "writer"))

        writer = SummaryWriter(join(logging_path, "writer"))
        writer.add_text("configs", str(p.format_values()))

        # 3. Optimizer
        optimizer = optim.Adam(classifier.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

        start_time = time()

        earlystop_counter = 0
        best_val_loss = None
        best_val_acc = None

        for epoch_idx in range(num_epochs):
            print("[Epoch {}]".format(epoch_idx))

            running_loss = 0.0
            running_acc = 0.0

            classifier.train()
            train_batch_generator = datasets.generate_batches_col(train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         drop_last=True,
                                                         device=device)
            # DEBUG
            #   weights = list(classifier.encoders['char'].linear1.parameters())[0]
            #   print("[DEBUG] Char encoder weights mean, max, min: {} {} {}".format(
            #   weights.mean(), weights.max(), weights.min()))

            for batch_idx, batch_dict in tqdm(enumerate(train_batch_generator)):
                y = batch_dict["label"]
                X = batch_dict["data"]

                optimizer.zero_grad()
                y_pred = classifier(X)

                # Calc loss
                loss = loss_func(y_pred, y)

                # Calc accuracy
                _, y_pred_ids = y_pred.max(1)
                acc = (y_pred_ids == y).sum().item() / batch_size

                # Update parameters
                loss.backward()
                optimizer.step()

                running_loss += (loss - running_loss) / (batch_idx + 1)
                running_acc += (acc - running_acc) / (batch_idx + 1)

            print("[Train] loss: {}".format(running_loss))
            print("[Train] acc: {}".format(running_acc))
            writer.add_scalar("train_loss", running_loss, epoch_idx)
            writer.add_scalar("train_acc", running_acc, epoch_idx)


            # Validation
            running_val_loss = 0.0
            running_val_acc = 0.0

            classifier.eval()

            with torch.no_grad():
                y_pred, y_true = [], []
                val_batch_generator = datasets.generate_batches_col(val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           drop_last=True,
                                                           device=device)
                for batch_idx, batch_dict in enumerate(val_batch_generator):
                    y = batch_dict["label"]
                    X = batch_dict["data"]

                    # Pred
                    pred = classifier(X)

                    y_pred.extend(pred.cpu().numpy())
                    y_true.extend(y.cpu().numpy())

                    # Calc loss
                    loss = loss_func(pred, y)

                    # Calc accuracy
                    _, pred_ids = torch.max(pred, 1)
                    acc = (pred_ids == y).sum().item() / batch_size

                    running_val_loss += (loss - running_val_loss) / (batch_idx + 1)
                    running_val_acc += (acc - running_val_acc) / (batch_idx + 1)

            print("[Val] loss: {}".format(running_val_loss))
            print("[Val] acc: {}".format(running_val_acc))
            writer.add_scalar("val_loss", running_val_loss, epoch_idx)
            writer.add_scalar("val_acc", running_val_acc, epoch_idx)

            # save prediction at each epoch
            #np.save(join(logging_path, "outputs", 'y_pred_epoch_{}.npy'.format(epoch_idx)), y_pred)
            #if epoch_idx == 0:
            #    np.save(join(logging_path, "outputs", 'y_true.npy'), y_true)

            if best_val_acc is None or running_val_acc > best_val_acc:
                best_acc_epoch = epoch_idx
                best_val_acc = running_val_acc

            # Early stopping
            if best_val_loss is None or running_val_loss < best_val_loss:
                best_loss_epoch = epoch_idx
                best_val_loss = running_val_loss
                earlystop_counter = 0
            else:
                earlystop_counter += 1

            if earlystop_counter >= patience:
                print("Warning: validation loss has not been improved more than {} epochs. Invoked early stopping.".format(patience))
                break

        writer.close()

        end_time = time()
        print("Training {}th of {} exps (with validation) done. ({} sec)".format(k+1, num_exp, int(end_time - start_time)))
        time_record['Train+validate'] = (end_time - start_time)

        #val_result['cv'].append(k + 1)
        #val_result['acc'].append(best_val_acc)
        #val_result['best_acc_epoch'].append(best_acc_epoch)
        #val_result['best_loss_epoch'].append(best_loss_epoch)

        if best_acc is None or best_val_acc > best_acc:
            best_acc = best_val_acc
            best_cv = k+1
            model_for_save = classifier

        #val_result_df = pd.DataFrame(val_result)
        #val_result_df.to_csv(join(logging_path, "validation{}.csv".format(cross_validation)))

        print("CV result : Best accuracy {} from {}th exp".format(best_acc, best_cv, num_exp))

        ################
        ##### test #####
        ################

        start_time = time()
        classifier.eval()

        running_test_loss = 0.0
        running_test_acc = 0.0
        with torch.no_grad():
            y_pred, y_true = [], []
            test_batch_generator = datasets.generate_batches_col(test_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           drop_last=True,
                                                           device=device)
            for batch_idx, batch_dict in enumerate(test_batch_generator):
                y = batch_dict["label"]
                X = batch_dict["data"]

                # Pred
                pred = classifier(X)

                y_pred.extend(pred.cpu().numpy())
                y_true.extend(y.cpu().numpy())

                # Calc loss
                loss = loss_func(pred, y)

                # Calc accuracy
                _, pred_ids = torch.max(pred, 1)
                acc = (pred_ids == y).sum().item() / batch_size

                running_test_loss += (loss - running_test_loss) / (batch_idx + 1)
                running_test_acc += (acc - running_test_acc) / (batch_idx + 1)
            print("[test] loss: {}".format(running_test_loss))
            print("[test] acc: {}".format(running_test_acc))

            report = classification_report(y_true, np.argmax(y_pred, axis=1), output_dict=True)
            result_list.append([k, report['macro avg']['f1-score'], report['weighted avg']['f1-score']])
            print("macro avg : {}, weighted avg : {}".format(report['macro avg']['f1-score'], report['weighted avg']['f1-score']))

        #print(df)
        #print("y_true : ", y_true, ", y_pred : ", np.argmax(y_pred, axis=1))
        #result = {'y_true': y_true, 'y_pred': np.argmax(y_pred, axis=1)}
        result = {'y_true': label_enc.inverse_transform(y_true), 'y_pred': label_enc.inverse_transform(np.argmax(y_pred, axis=1))}
        result_df = pd.DataFrame(result)
        result_df.to_csv(join(logging_path, "prediction{}.csv".format(k+1)))

        end_time = time()
        print("Evaluation time {} sec.".format(int(end_time - start_time)))
        time_record['Evaluation'] = (end_time - start_time)

    print("logging_path : {}".format(logging_path))
    df = pd.DataFrame(result_list, columns=['CV', 'macro_avg', 'weighted_avg'])
    df.to_csv(join(logging_path, "cv_results.csv"))

    print("results saved")

    #######################################################################################################
    ########################################## output model name ##########################################
    #######################################################################################################

    print("Saving model...")

    #torch.save(model_for_save.state_dict(), join(logging_path, "model.pt"))
    # save as pretrained model
    pre_trained_loc = join(os.environ['BASEPATH'], 'model', 'pre_trained_sherlock', TYPENAME)
    if not os.path.exists(pre_trained_loc):
        os.makedirs(pre_trained_loc)

    pretrained_name = '{}.pt'.format(logging_name) if cross_validation is None else \
        '{}_{}-fold.pt'.format(logging_name, cross_validation)

    torch.save(model_for_save.state_dict(), join(pre_trained_loc, pretrained_name))

    print("model saved")



