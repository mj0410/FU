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
    p.add('--n_worker', type=int, default=4, help='# of workers for dataloader')
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
    p.add('--mode', type=str, help='experiment mode', choices=['train', 'eval'], default='train')
    p.add('--model_list', nargs='+', type=str, help='For eval mode only, load pretrained models')
    p.add('--train_percent', type=str, default='train', help='Training with only part of the data, post-fix in the train-split file.')

    p.add('--cross_validation', type=int_or_none, default=None)

    # only implemented for cross validation, each patition has full/ multi-col version
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
        LDANAME = os.environ['LDA_name']
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
    t1 = time()
    print("Creating Dataset object...")
    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    train_test_path = join(os.environ['BASEPATH'], 'extract', 'out', 'train_test_split')
    train_list, val_list, test_list = [], [], []

    split = pd.read_csv(join(train_test_path, '{}.csv'.format(train_test_file)))

    if cross_validation is None:
        train = split[split['split'] == 'train']#.reset_index(drop=True) #[:10]
        val = split[split['split'] == 'val']#.reset_index(drop=True) #[:10]
        test = split[split['split'] == 'test']#.reset_index(drop=True) #[:10]
    else:
        # just for extract number of 'k' of k-fold
        a, b = train_test_file.split('fold')
        k = a[-1]
        print('Conducting cross validation, current {}th experiment in {}-fold CV'.format(cross_validation, k))

        val = split[split['cv'] == cross_validation]
        test = split[split['cv'] == 'test']
        train = split[(split['cv'] != 'test') | (split['cv'] != cross_validation)]

    print('data length:')
    print(len(train), len(val), len(test))

    tableFeatures = datasets.TableFeatures(header_file,
                                           sherlock_feature_file,
                                           sherlock_feature_groups,
                                           extracted_feature_topic=topic_feature_file,
                                           topic_feature=topic_name,
                                           label_enc=label_enc,
                                           id_filter=None,
                                           max_col_count=None)

    if args.mode != 'eval':
        train_tf = copy.copy(tableFeatures).set_filter(train['dataset_id'], train['row_index']).to_col()
        train_list.append(train_tf)

        val_tf = copy.copy(tableFeatures).set_filter(val['dataset_id'], val['row_index']).to_col()
        val_list.append(val_tf)

    test_tf = copy.copy(tableFeatures).set_filter(test['dataset_id'], test['row_index']).to_col()
    test_list.append(test_tf)

    if args.mode!='eval':
        train_dataset = ConcatDataset(train_list)
        val_dataset = ConcatDataset(val_list)
    test_dataset = ConcatDataset(test_list)

    t2 = time()
    print("Done ({} sec.)".format(int(t2 - t1)))
    time_record['Load'] = (t2 - t1)

    # 2. Models

    start_time = time()

    classifier = build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim, dropout_ratio=dropout_ratio).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)

    if args.mode == 'train':
        writer = SummaryWriter(logging_path)
        writer.add_text("configs", str(p.format_values()))


        # 3. Optimizer
        optimizer = optim.Adam(classifier.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
        
        earlystop_counter = 0
        best_val_loss = None
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
    #        weights = list(classifier.encoders['char'].linear1.parameters())[0]
    #        print("[DEBUG] Char encoder weights mean, max, min: {} {} {}".format(
    #            weights.mean(), weights.max(), weights.min()))

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
                    # print("X : ", X)
                    #print("y : ", y)

                    # Pred
                    pred = classifier(X)
                    #print("val_pred : {}".format(pred))

                    y_pred.extend(pred.cpu().numpy())
                    y_true.extend(y.cpu().numpy())

                    # Calc loss
                    loss = loss_func(pred, y)

                    # Calc accuracy
                    _, pred_ids = torch.max(pred, 1)
                    #print("pred_ids : ", pred_ids)
                    acc = (pred_ids == y).sum().item() / batch_size

                    running_val_loss += (loss - running_val_loss) / (batch_idx + 1)
                    running_val_acc += (acc - running_val_acc) / (batch_idx + 1)

            print("[Val] loss: {}".format(running_val_loss))
            print("[Val] acc: {}".format(running_val_acc))
            writer.add_scalar("val_loss", running_val_loss, epoch_idx)
            writer.add_scalar("val_acc", running_val_acc, epoch_idx)

            if not os.path.exists(join(logging_path, "outputs")):
                os.makedirs(join(logging_path, "outputs"))

            # save prediction at each epoch
            np.save(join(logging_path, "outputs", 'y_pred_epoch_{}.npy'.format(epoch_idx)), y_pred)
            if epoch_idx == 0:
                np.save(join(logging_path, "outputs", 'y_true.npy'), y_true)

            
            # Early stopping
            if best_val_loss is None or running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                earlystop_counter = 0
            else:
                earlystop_counter += 1
            
            if earlystop_counter >= patience:
                print("Warning: validation loss has not been improved more than {} epochs. Invoked early stopping.".format(patience))
                break

        #print("classifier", classifier)
        print("Saving model...")

        #######################################################################################################
        ########################################## output model name ##########################################
        #######################################################################################################

        torch.save(classifier.state_dict(),join(logging_path, "model.pt"))
        # save as pretrained model
        pre_trained_loc = join(os.environ['BASEPATH'],'model','pre_trained_sherlock', TYPENAME)
        if not os.path.exists(pre_trained_loc):
                os.makedirs(pre_trained_loc)

        pretrained_name = '{}.pt'.format(logging_name) if cross_validation is None else\
                          '{}_{}-fold_{}.pt'.format(logging_name, k, cross_validation)


        torch.save(classifier.state_dict(), join(pre_trained_loc, pretrained_name))

        writer.close()

        end_time = time()
        print("Training (with validation) ({} sec.)".format(int(end_time - start_time)))
        time_record['Train+validate'] = (end_time - start_time)


    elif args.mode == 'eval':
        start_time = time()
        # load pre-trained model
        result_list = []
        model_loc = join(os.environ['BASEPATH'],'model','pre_trained_sherlock', TYPENAME)
        for model_path in args.model_list:
            classifier.load_state_dict(torch.load(join(model_loc, model_path), map_location=device))
            classifier.eval()

            # eval
            running_test_loss = 0.0
            running_test_acc = 0.0
            with torch.no_grad():
                y_pred, y_true = [], []
                test_batch_generator = datasets.generate_batches_col(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       device=device)
                for batch_idx, batch_dict in enumerate(test_batch_generator):
                    y = batch_dict["label"]
                    print(y)
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

                #print(np.array(y_true).shape, np.array(y_pred).shape)
                #np.argmax(y_pred, axis=1)
                #print(np.argmax(y_pred, axis=1))
                report = classification_report(y_true, np.argmax(y_pred, axis=1), output_dict=True)
                #print(report)

                #print(report['macro avg'], report['weighted avg'])
                result_list.append([model_path, report['macro avg']['f1-score'], report['weighted avg']['f1-score']])

        df = pd.DataFrame(result_list, columns=['model', 'macro avg', 'weighted avg'])
        print(df)
        #print("y_true : ", y_true, ", y_pred : ", np.argmax(y_pred, axis=1))
        result = {'y_true': y_true, 'y_pred': np.argmax(y_pred, axis=1)}
        result_df = pd.DataFrame(result)
        print(result_df.head())
        print("logging_path : {}".format(logging_path))
        result_df.to_csv(join(logging_path, "prediction{}.csv".format(cross_validation)))

        end_time = time()
        print("Evaluation time {} sec.".format(int(end_time - start_time)))
        time_record['Evaluation'] = (end_time - start_time)
        # save

    # save time
    with open(join(logging_path, "outputs", 'time.json'), 'w') as f:
        json.dump(time_record, f)



