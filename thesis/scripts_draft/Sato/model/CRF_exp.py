import os
import sys
from os.path import join
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)
import time, random
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import numpy.ma as ma
import datetime
import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import LabelEncoder

from model import models_sherlock, datasets

from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
import itertools
from torchcrf import CRF
import configargparse
import copy
from utils import get_valid_types, str2bool, str_or_none, name2dic, int_or_none
from multiprocessing import Pool, freeze_support, RLock


#################### 
# get col_predictions as emission and use pytorchcrf package.
#################### 

RANDSEED = 10000
prng = np.random.RandomState(RANDSEED)
#################### 
# Load configs
#################### 

if __name__ == "__main__":
    freeze_support()

    p = configargparse.ArgParser()
    p.add('-c', '--config_file', required=True, is_config_file=True, help='config file path')

    # general configs
    p.add('--MAX_COL_COUNT', type=int, default=6, help='Max number of columns in a table (padding for batches)')
    p.add('--table_batch_size', type=int, default=100, help='# of tables in a batch')
    p.add('--n_worker', type=int, default=4, help='# of workers for dataloader')
    p.add('--TYPENAME', type=str, help='Name of valid types', env_var='TYPENAME')
    p.add('--train_test_split', type=str, help='Name of output from split_train_test.py')
    p.add('--header', type=str, help='Name of output from extract_header.py')
    p.add('--feature', type=str, help='Name of output from extract_feature.py')

    # NN configs, ignored in evaluation mode
    p.add('--epochs', type=int, default=10)
    p.add('--learning_rate', type=float, default=0.01)
    p.add('--decay', type=float, default=1e-4)
    p.add('--dropout_rate', type=float, default=0.35)
    p.add('--optimizer_type', type=str, default='adam')
    p.add('--patience', type=int, default=100, help='patience for early stopping')

    # sherlock configs
    p.add('--sherlock_feature_groups', default=['char','rest','par','word'])
    p.add('--topic', type=str_or_none, default=None)
    p.add('--pre_trained_sherlock_path', type=str, default='None.pt')
    p.add('--fixed_sherlock_params', type=str2bool, default=True)

    # exp configs
    p.add('--init_matrix_path', type=str_or_none, default=None)
    p.add('--training_acc', type=str2bool, default=True, help='Calculate training accuracy (in addition to loss) for debugging')
    p.add('--shuffle_col', type=str2bool, default=False, help='Shuffle the columns in tables while training the model')

    p.add('--cross_validation', type=int_or_none, default=None, help='Format CVn-k, load the kth exp for n-fold cross validation')
    # load train test from extract/out/train_test_split/CVn_{}.json, kth exp hold kth partition for evaluation
    # save output and model file with postfix CVn-k
    # if set to none, use standard parition from the train_test_split files.train_test_split
    p.add('--multi_col_eval', type=str2bool, default=False, help='Evaluate using only multicol, train using all ')
    # only implemented for cross validation, each patition has full/ multi-col version

    p.add('--num', type=int, default=100, help='number of experiments')

    p.add('--comment', type=str, default='')

    args = p.parse_args()

    print("----------")
    print(args)
    print("----------")
    print(p.format_values())    # useful for logging where different settings came from
    print("----------")

    # general configs
    MAX_COL_COUNT = args.MAX_COL_COUNT
    batch_size = args.table_batch_size
    n_worker = args.n_worker
    TYPENAME = args.TYPENAME
    train_test_file = args.train_test_split
    header_file = args.header
    feature_file = args.feature
    # NN configs
    epochs = args.epochs
    learning_rate = args.learning_rate
    decay = args.decay
    dropout_rate = args.dropout_rate
    ## sherlock configs
    sherlock_feature_groups = args.sherlock_feature_groups
    topic = args.topic
    pre_trained_sherlock_path = args.pre_trained_sherlock_path
    fixed_sherlock_params = args.fixed_sherlock_params
    # exp configs
    init_matrix_path = args.init_matrix_path
    training_acc= args.training_acc
    shuffle_col = args.shuffle_col
    shuffle_seed = 10
    patience = args.patience

    cross_validation = args.cross_validation
    num_exp = args.num

    ####################
    # Preparations
    ####################
    valid_types = get_valid_types(TYPENAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    if topic:
        model_loc = os.environ['TOPICMODELPATH']
        kwargs_file = join(model_loc, topic, "{}.pkl".format(topic))
        with open(kwargs_file, 'rb') as f:
            kwargs = pickle.load(f)
        topic_dim = kwargs['tn']
    else:
        topic_dim = None

    ## load single column model
    pre_trained_sherlock_loc = join(os.environ['BASEPATH'],'model','pre_trained_sherlock', TYPENAME)
    classifier = models_sherlock.build_sherlock(sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim, dropout_ratio=dropout_rate).to(device)

    # fix sherlock parameters
    if fixed_sherlock_params:
        for name, param in classifier.named_parameters():
            param.requires_grad = False

    ## Label encoder


    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    # initialize with co-coccur matrix
    if init_matrix_path is None:
        print("Using random initial transitions")
        L = len(valid_types)
        init_transition = prng.rand(L, L)
    else:
        init_matrix_loc = join(os.environ['BASEPATH'], 'model', 'co_occur_matrix')
        matrix_co = np.load(join(init_matrix_loc, init_matrix_path))
        init_transition = np.log(matrix_co+1)

    # fix random seed for reproducibility
    if shuffle_col:
        torch.manual_seed(shuffle_seed)
        if device == torch.device('cuda'):
            torch.cuda.manual_seed_all(shuffle_seed)



    # tensorboard logger
    currentDT = datetime.datetime.now()
    DTString = '-'.join([str(x) for x in currentDT.timetuple()[:5]])
    logging_base = 'CRF_log' #if device == torch.device('cpu') else 'CRF_cuda_log'
    #logging_path = join(os.environ['BASEPATH'],'results', logging_base, TYPENAME, '{}_{}_{}'.format(config_name, args.comment, DTString))

    logging_name = '{}'.format(args.comment)

    logging_path = join(os.environ['BASEPATH'],'results', logging_base, TYPENAME, logging_name)

    time_record = {}

    ####################
    # Helpers
    ####################

    # evaluate and return prediction & true labels of a table batch
    def eval_batch(table_batch, label_batch, mask_batch):
        # reshap (table_batch * table_size * features)
        for f_g in table_batch:
            table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

        emissions = classifier(table_batch).view(batch_size, MAX_COL_COUNT, -1)
        pred = model.decode(emissions, mask_batch)

        pred = np.concatenate(pred)
        labels = label_batch.view(-1).cpu().numpy()
        masks = mask_batch.view(-1).cpu().numpy()
        invert_masks = np.invert(masks==1)

        return pred, ma.array(labels, mask=invert_masks).compressed()

    # randomly shuffle the orders of columns in a table batch
    def shuffle_column(table_batch, label_batch, mask_batch):
        batch_size = label_batch.shape[0]
        for b in range(batch_size):
            mask= mask_batch[b]
            valid_length = mask.sum()
            new_order = torch.cat((torch.randperm(valid_length), torch.arange(valid_length,len(mask))))

            for f_g in table_batch:
                table_batch[f_g][b] = table_batch[f_g][b][new_order]
            label_batch[b] = label_batch[b][new_order]

        return table_batch, label_batch, mask_batch

    remove_support = lambda dic: {i:dic[i] for i in dic if i!='support'}


    ####################
    # Load data
    ####################

    train_test_path = join(os.environ['BASEPATH'], 'extract', 'out', 'train_test_split')
    train_list, val_list, test_list = [], [], []

    split = pd.read_csv(join(train_test_path, '{}.csv'.format(train_test_file)))

    tableFeatures = datasets.TableFeatures(header_file=header_file,
                                           sherlock_features=sherlock_feature_groups,
                                           extracted_feature_topic=feature_file,
                                           topic_feature=topic,
                                           label_enc=label_enc,
                                           id_filter=None,
                                           max_col_count=MAX_COL_COUNT)

    # 2. Models
    result_list = []
    best_acc = None
    print("run {} experiments".format(num_exp))

    for k in range(num_exp):

        writer = SummaryWriter(logging_path)
        writer.add_text("configs", str(p.format_values()))

        print('Conducting cross validation, current {}th in {} experiments'.format(k + 1, num_exp))

        print("Creating Dataset object...")
        Dataset_start_time = time.time()

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

        train_tf = copy.copy(tableFeatures).set_filter(train['dataset_id'], train['row_index'])
        train_list.append(train_tf)

        train_end_time = time.time()
        print("Train datasets are created. ({} sec)".format(int(train_end_time - Dataset_start_time)))

        val_tf = copy.copy(tableFeatures).set_filter(val['dataset_id'], val['row_index'])
        val_list.append(val_tf)

        val_end_time = time.time()
        print("Val datasets are created. ({} sec)".format(int(val_end_time - train_end_time)))

        test_tf = copy.copy(tableFeatures).set_filter(test['dataset_id'], test['row_index'])
        test_list.append(test_tf)

        test_end_time = time.time()
        print("Test datasets are created. ({} sec)".format(int(test_end_time - val_end_time)))

        train_dataset = ConcatDataset(train_list)
        val_dataset = ConcatDataset(val_list)
        test_dataset = ConcatDataset(test_list)

        Dataset_end_time = time.time()
        print("Dataset objects are created. ({} sec)".format(int(Dataset_end_time - Dataset_start_time)))

        model = CRF(len(valid_types) , batch_first=True).to(device)

        print("model training...")

        classifier.load_state_dict(torch.load(join(pre_trained_sherlock_loc, pre_trained_sherlock_path), map_location=device))
        print("{} loaded".format(pre_trained_sherlock_path))

        # Set initial transition parameters
        if init_transition is not None:
            model.transitions = torch.nn.Parameter(torch.tensor(init_transition).float().to(device))

        if fixed_sherlock_params:
            param_list = list(model.parameters())
        else:
            # learnable sherlock features, but with a small fixed learning rate.
            param_list = [{'params': model.parameters()} , {'params':classifier.parameters(), 'lr':1e-4}]


        if args.optimizer_type=='sgd':
            optimizer = optim.SGD(param_list, lr=learning_rate, weight_decay=decay)
        elif args.optimizer_type=='adam':
            optimizer = optim.Adam(param_list, lr=learning_rate, weight_decay=decay)
        else:
            assert False, "Unsupported optimizer type"

        ####################
        # Get baseline accuracy
        ####################

        with torch.no_grad():
            classifier.eval()
            y_pred, y_true = [], []
            train_batch = datasets.generate_batches(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=True,
                                                       device=device,
                                                       n_workers=n_worker)

            for table_batch, label_batch, mask_batch in tqdm(train_batch, desc='Single col Accuracy'):
                for f_g in table_batch:
                    table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

                pred_scores = classifier.predict(table_batch)#.view(batch_size, MAX_COL_COUNT, -1)
                pred = torch.argmax(pred_scores, dim=1).cpu().numpy()

                labels = label_batch.view(-1).cpu().numpy()
                masks = mask_batch.view(-1).cpu().numpy()
                invert_masks = np.invert(masks==1)

                y_pred.extend(ma.array(pred, mask=invert_masks).compressed())
                y_true.extend(ma.array(labels, mask=invert_masks).compressed())

            #print("y_true : ", y_true)
            #print("y_pred : ", y_pred)
            val_acc = classification_report(y_true, y_pred, output_dict=True)
            print('[BASELINE]')
            print("[Col val acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'], val_acc['weighted avg']['f1-score']))

            writer.add_scalars('marco avg-col', remove_support(val_acc['macro avg']), k)
            writer.add_scalars('weighted avg-col', remove_support(val_acc['weighted avg']), k)

            #np.save(join(logging_path, "outputs", 'y_pred_sherlock.npy'), label_enc.inverse_transform(y_pred))
            #np.save(join(logging_path, "outputs", 'y_true.npy'), label_enc.inverse_transform(y_true))



        start_time = time.time()

        loss_counter = 0
        best_val_loss = None
        best_val_acc = None

        for epoch_idx in range(epochs):
            print("[Epoch {}/{}] ============================".format(epoch_idx,epochs))

            running_loss = 0.0
            running_acc = 0.0

            # set single col prediciton to eval mode
            model.train()
            if fixed_sherlock_params:
                classifier.eval()
            else:
                classifier.train()

            training = datasets.generate_batches(train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         drop_last=True,
                                                         device=device,
                                                         n_workers=n_worker)
            it = 0
            accumulate_loss = 0.0
            label_list = None

            training_iter = tqdm(training, desc="Training")
            for table_batch, label_batch, mask_batch in training_iter:

                if shuffle_col:
                    table_batch, label_batch, mask_batch = shuffle_column(table_batch, label_batch, mask_batch)
                    if k==0 and epoch_idx==0:
                        if label_list is None:
                            label_list = []
                        label_list.append(label_batch.tolist())

                # Step1. Clear gradient
                optimizer.zero_grad()
                for f_g in table_batch:
                    table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

                emissions = classifier(table_batch).view(batch_size, MAX_COL_COUNT, -1).to(device)


                # Step 2. Run forward pass.
                loss = -model(emissions, label_batch, mask_batch, reduction='mean').to(device)

                # Step 3. Compute the loss, gradients, and update the parameters
                loss.backward()
                optimizer.step()

                accumulate_loss += loss.item()
                it +=1
                if it %500 ==1:
                    training_iter.set_postfix(loss = (accumulate_loss/(it)))
                    writer.add_scalar("val_loss", accumulate_loss/(it), loss_counter)
                    loss_counter += 1

            epoch_loss = accumulate_loss/(it)
            writer.add_scalar("epoch_train_loss", epoch_loss, epoch_idx)

            if shuffle_col and label_list is not None:
                print(it, len(label_list))
                with open(join(logging_path, 'shuffled_label.txt'), 'w') as f:
                    for l in label_list:
                        f.write(str(l))
                        f.write('\n')

            # Training accuracy
            # could be omitted
            if training_acc:
                y_pred, y_true = [], []
                with torch.no_grad():
                    model.eval()
                    classifier.eval()
                    training_batch = datasets.generate_batches(train_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=False,
                                                               drop_last=True,
                                                               device=device,
                                                               n_workers=n_worker)

                    for table_batch, label_batch, mask_batch in tqdm(training_batch, desc='Training Accuracy'):
                        pred, labels = eval_batch(table_batch, label_batch, mask_batch)
                        y_pred.extend(pred)
                        y_true.extend(labels)

                    train_acc = classification_report(y_true, y_pred, output_dict=True)
                    writer.add_scalars('marco avg-train', remove_support(train_acc['macro avg']), epoch_idx)
                    writer.add_scalars('weighted avg-train', remove_support(train_acc['weighted avg']), epoch_idx)

            # Validation accuracy
            y_pred, y_true = [], []
            final_val_loss = 0.0
            iter = 0
            with torch.no_grad():
                iter += 1
                model.eval()
                classifier.eval()
                val_batch = datasets.generate_batches(val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           drop_last=True,
                                                           device=device,
                                                           n_workers=n_worker)

                it = 0
                running_val_loss = 0.0
                for table_batch, label_batch, mask_batch in tqdm(val_batch, desc='Validation Accuracy'):
                    pred, labels = eval_batch(table_batch, label_batch, mask_batch)

                    emissions_val = classifier(table_batch).view(batch_size, MAX_COL_COUNT, -1).to(device)

                    val_loss = -model(emissions_val, label_batch, mask_batch, reduction='mean').to(device)

                    running_val_loss += val_loss.item()
                    it += 1

                    y_pred.extend(pred)
                    y_true.extend(labels)
                #print("y_true : {}, y_pred : {}".format(y_true, y_pred))
                accumulate_val_loss = running_val_loss / it
                final_val_loss += accumulate_val_loss
                val_acc = classification_report(y_true, y_pred, output_dict=True)
                writer.add_scalars('marco avg-val', remove_support(val_acc['macro avg']), epoch_idx)
                writer.add_scalars('weighted avg-val', remove_support(val_acc['weighted avg']), epoch_idx)

            # printing stats
            print("[Train loss]: {}".format(epoch_loss))
            if training_acc:
                print("[Train acc]: marco avg F1 {}, weighted avg F1 {}".format(train_acc['macro avg']['f1-score'], train_acc['weighted avg']['f1-score']))
            print("[Val   acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'], val_acc['weighted avg']['f1-score']))

            running_val_acc = val_acc['weighted avg']['f1-score']
            epoch_val_loss = final_val_loss / iter
            print("val_loss : ", epoch_val_loss)

            if best_val_acc is None or running_val_acc > best_val_acc:
                best_acc_epoch = epoch_idx
                best_val_acc = running_val_acc

            if best_val_loss is None or epoch_val_loss < best_val_loss:
                best_loss_epoch = epoch_idx
                best_val_loss = epoch_val_loss
                earlystop_counter = 0
            else:
                earlystop_counter += 1

            if earlystop_counter >= patience:
                print(
                    "Warning: validation loss has not been improved more than {} epochs. Invoked early stopping.".format(
                        patience))
                break

        writer.close()

        end_time = time.time()

        print("Training (with validation) ({} sec.)".format(int(end_time - start_time)))
        time_record['Train+validate'] = (end_time - start_time)

        if best_acc is None or best_val_acc > best_acc:
            best_acc = best_val_acc
            best_cv = k+1
            classifier_for_save = classifier
            model_for_save = model

        print("CV result : Best accuracy {} from {}th exp".format(best_acc, best_cv, num_exp))

        ##################################################################
        ######################### Evaluation #############################
        ##################################################################

        start_time = time.time()

        classifier_for_save.eval()
        model_for_save.eval()

        with torch.no_grad():
            y_pred, y_true = [], []
            test_batch = datasets.generate_batches(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=True,
                                                       device=device,
                                                       n_workers=n_worker)

            for table_batch, label_batch, mask_batch in tqdm(test_batch, desc='Single col Accuracy'):
                for f_g in table_batch:
                    table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

                pred_scores = classifier_for_save.predict(table_batch)  # .view(batch_size, MAX_COL_COUNT, -1)
                pred = torch.argmax(pred_scores, dim=1).cpu().numpy()

                labels = label_batch.view(-1).cpu().numpy()
                masks = mask_batch.view(-1).cpu().numpy()
                invert_masks = np.invert(masks == 1)

                y_pred.extend(ma.array(pred, mask=invert_masks).compressed())
                y_true.extend(ma.array(labels, mask=invert_masks).compressed())

            val_acc = classification_report(y_true, y_pred, output_dict=True)
            print('[Single-Col model (Maybe fine-tuned)]')
            print("[Col val acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'],
                                                                                      val_acc['weighted avg'][
                                                                                          'f1-score']))

                #        # save sherlock predictions
                #        if not os.path.exists(join(logging_path, "outputs")):
                #            os.makedirs(join(logging_path, "outputs"))#

                #        np.save(join(logging_path, "outputs", 'y_pred_sherlock.npy'), label_enc.inverse_transform(y_pred))
                #        np.save(join(logging_path, "outputs", 'y_true.npy'), label_enc.inverse_transform(y_true))
                # Validation accuracy

        y_pred, y_true = [], []
        with torch.no_grad():
            model_for_save.eval()
            classifier_for_save.eval()
            test_batch = datasets.generate_batches(test_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           drop_last=True,
                                                           device=device,
                                                           n_workers=n_worker)

            for table_batch, label_batch, mask_batch in tqdm(test_batch, desc='Validation Accuracy'):
                pred, labels = eval_batch(table_batch, label_batch, mask_batch)

                y_pred.extend(pred)
                y_true.extend(labels)

            val_acc = classification_report(y_true, y_pred, output_dict=True)
            print('[Model]')
            print("[Model val acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'],
                                                                                      val_acc['weighted avg'][
                                                                                          'f1-score']))

            result_list.append([k, val_acc['macro avg']['f1-score'], val_acc['weighted avg']['f1-score']])

        #result = {'y_true': y_true, 'y_pred': y_pred}
        result = {'y_true': label_enc.inverse_transform(y_true), 'y_pred': label_enc.inverse_transform(y_pred)}

        result_df = pd.DataFrame(result)
        result_df.to_csv(join(logging_path, "prediction{}.csv".format(k + 1)))

        end_time = time.time()
        print("Evaluation time {} sec.".format(int(end_time - start_time)))
        time_record['Evaluation'] = (end_time - start_time)

    print("logging_path : {}".format(logging_path))
    df = pd.DataFrame(result_list, columns=['CV', 'macro_avg', 'weighted_avg'])
    df.to_csv(join(logging_path, "cv_results.csv"))

    print("results saved")

    pre_trained_loc = join(os.environ['BASEPATH'],'model','pre_trained_CRF', TYPENAME)
    if not os.path.exists(pre_trained_loc):
            os.makedirs(pre_trained_loc)

    pretrained_name = '{}.pt'.format(logging_name) if cross_validation is None else \
        '{}_{}-fold.pt'.format(logging_name, cross_validation)

    torch.save({'col_classifier': classifier_for_save.state_dict() ,
                    'CRF_model': model_for_save.state_dict()}
                    ,join(pre_trained_loc, pretrained_name))

    print("models saved")

