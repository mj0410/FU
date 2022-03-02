import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from os.path import join
from os import listdir
import configargparse
import plotinpy as pnp

TYPENAME = os.environ['TYPENAME']
BASEPATH = os.environ['BASEPATH']

def valid_types(TYPENAME):
    with open(join(os.environ['RESULT'], 'types.json'), 'r') as typefile:
        valid_types = json.load(typefile)[TYPENAME]
    return valid_types


header_path = join(BASEPATH, 'extract', 'out', 'headers')
split_path = join(BASEPATH, 'extract', 'out', 'train_test_split')
#output_path = join(BASEPATH, 'results', 'analysis')
output_path = join(os.environ['RESULT'], 'analysis')
result_path = join(os.environ['RESULT'], 'results')

valid_types = valid_types(TYPENAME)


def number_of_data(file):

    field_count = [0] * len(valid_types)
    print("number of tables : ", len(file))

    table_names = file.dataset_id.unique()

    for id in table_names:
        print(id)
        df_id = file[file['dataset_id'] == id]
        file_path = df_id.iloc[0]['locator']
        df = pd.read_csv(join(file_path, id))

        row_index = []
        for i in range(len(df_id)):
            row_index += eval(df_id.iloc[i]['row_index'])

        df = df.loc[row_index]
        field_list = eval(df_id.iloc[0]['field_list'])
        field_name = eval(df_id.iloc[0]['field_names'])

        for i in range(len(field_list)):
            num_values = df.iloc[:, [i]].dropna()
            field_count[field_name[i]] += len(num_values)

    df_dict = {'field_name': valid_types, 'number_of_data': field_count}
    df = pd.DataFrame(df_dict)
    df = df.sort_values(by=['number_of_data'], ascending=False)

    return df

def data_for_violinplot(sherlock, topic):
  macro_sherlock = {'acc_type': ['macro']*len(sherlock), 'accuracy': sherlock['macro_avg'], 'model': ['sherlock']*len(sherlock)}
  weighted_sherlock = {'acc_type': ['weighted']*len(sherlock), 'accuracy': sherlock['weighted_avg'], 'model': ['sherlock']*len(sherlock)}
  macro_topic = {'acc_type': ['macro']*len(topic), 'accuracy': topic['macro_avg'], 'model': ['CRF']*len(topic)}
  weighted_topic = {'acc_type': ['weighted']*len(topic), 'accuracy': topic['weighted_avg'], 'model': ['CRF']*len(topic)}

  dicts = [macro_sherlock, weighted_sherlock, macro_topic, weighted_topic]

  super_dict = {}
  for d in dicts:
    for k, v in d.items():
      super_dict.setdefault(k, []).extend(v)
  return super_dict

def data_for_violinplot3(sherlock, topic1, topic1_num, topic2, topic2_num):
  macro_sherlock = {'acc_type': ['macro']*len(sherlock), 'accuracy': sherlock['macro_avg'], 'model': ['sherlock']*len(sherlock)}
  weighted_sherlock = {'acc_type': ['weighted']*len(sherlock), 'accuracy': sherlock['weighted_avg'], 'model': ['sherlock']*len(sherlock)}
  macro_topic1 = {'acc_type': ['macro']*len(topic1), 'accuracy': topic1['macro_avg'], 'model': ['topic{}'.format(topic1_num)]*len(topic1)}
  weighted_topic1 = {'acc_type': ['weighted']*len(topic1), 'accuracy': topic1['weighted_avg'], 'model': ['topic{}'.format(topic1_num)]*len(topic1)}
  macro_topic2 = {'acc_type': ['macro'] * len(topic2), 'accuracy': topic2['macro_avg'], 'model': ['topic{}'.format(topic2_num)] * len(topic2)}
  weighted_topic2 = {'acc_type': ['weighted'] * len(topic2), 'accuracy': topic2['weighted_avg'], 'model': ['topic{}'.format(topic2_num)] * len(topic2)}

  dicts = [macro_sherlock, weighted_sherlock, macro_topic1, weighted_topic1, macro_topic2, weighted_topic2]

  super_dict = {}
  for d in dicts:
    for k, v in d.items():
      super_dict.setdefault(k, []).extend(v)
  return super_dict

def data_for_barplot(model, path):
    pred_acc = None
    prediction_results = [f for f in listdir(join(result_path, model, TYPENAME, path)) if f.startswith('prediction')]

    for file in prediction_results:
        pred = pd.read_csv(join(result_path, model, TYPENAME, path, file))
        pred['eval'] = pred['y_true'] == pred['y_pred']
        pred_mean = pred.groupby(['y_true']).eval.mean()
        if pred_acc is None:
            pred_acc = pd.DataFrame(pred_mean).reset_index()
            pred_acc = pred_acc.rename(columns={"eval": file})
        else:
            pred_tmp = pd.DataFrame(pred_mean).reset_index()
            pred_tmp = pred_tmp.rename(columns={"eval": file})
            pred_acc = pred_acc.join(pred_tmp.set_index('y_true'), on='y_true')
    pred_acc['avg_acc'] = pred_acc.mean(numeric_only=True, axis=1)
    pred_acc = pred_acc[['y_true', 'avg_acc']].rename(columns={"y_true": 'types'})
    print(pred_acc)

    return pred_acc

if __name__ == "__main__":

    p = configargparse.ArgParser()
    p.add('--split', type=str, default=None)
    p.add('--mode', nargs='+', type=str, help="number of input data : num, violin plot : violin2 or violin3, bar plot : bar, average accuracy : avg")
    p.add('--num_name', type=str, help='name of number of data plot')
    p.add('--violinplot_name', type=str, help='name of violinplot')
    p.add('--barplot_name', type=str, help='name of boxplot')

    args = p.parse_args()

    split_file = args.split
    mode = args.mode
    num_name = args.num_name
    violinplot_name = args.violinplot_name
    barplot_name = args.barplot_name

    s_model_name = os.environ['sherlock_model']
    st_model_name1 = os.environ['topic_model1']
    st_model_name2 = os.environ['topic_model2']
    crfs_model_name = os.environ['crf_sherlock_model']
    crft_model_name1 = os.environ['crf_topic_model1']
    crft_model_name2 = os.environ['crf_topic_model2']

    topic1_num = os.environ['topic1_num']
    topic2_num = os.environ['topic2_num']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if 'avg' in mode:

        model_name_sherlock_A = ['sherlock', 't9', 't18']
        model_name_sherlock_B = ['sherlock_syntheaB', 't9_syntheaB', 't18_syntheaB']

        for a in model_name_sherlock_A:
            try:
                df = pd.read_csv(join(BASEPATH, 'results', 'sherlock_log', 'syntheaA', a, 'cv_results.csv'))
                print('sherlock', a)
                print(df.mean())
                print('\n')
            except:
                print('There is no {}'.format(a))

        for b in model_name_sherlock_B:
            try:
                df = pd.read_csv(join(BASEPATH, 'results', 'sherlock_log', 'syntheaB', b, 'cv_results.csv'))
                print('sherlock', b)
                print(df.mean())
                print('\n')
            except:
                print('There is no {}'.format(b))

        model_name_CRF_A = ['CRF_sherlock', 'CRF_t9', 'CRF_t18']
        model_name_CRF_B = ['sherlock', 'CRF_rcol_t9', 'CRF_rcol_t18']

        for ca in model_name_CRF_A:
            try:
                df = pd.read_csv(join(BASEPATH, 'results', 'CRF_log', 'syntheaA', ca, 'cv_results.csv'))
                print('CRF', ca)
                print(df.mean())
                print('\n')
            except:
                print('There is no {}'.format(ca))

        for cb in model_name_CRF_B:
            try:
                df = pd.read_csv(join(BASEPATH, 'results', 'CRF_log', 'syntheaB', cb, 'cv_results.csv'))
                print('CRF', cb)
                print(df.mean())
                print('\n')
            except:
                print('There is no {}'.format(cb))

    if 'num' in mode:
        print("number of data")
        print("type : {} (number of types : {})".format(TYPENAME, len(valid_types)))

        split = pd.read_csv(join(split_path, '{}.csv'.format(split_file)))
        train = split[split['cv'] != 'test']
        test = split[split['cv'] == 'test']

        df_split = number_of_data(split)
        df_train = number_of_data(train)
        df_test = number_of_data(test)

        df_split.to_csv(join(output_path, "number_of_data_whole_{}.csv".format(TYPENAME)))
        df_train.to_csv(join(output_path, "number_of_data_train_{}.csv".format(TYPENAME)))
        df_test.to_csv(join(output_path, "number_of_data_test_{}.csv".format(TYPENAME)))

        fig = plt.figure(figsize=(20, 15))
        plot = sns.barplot(x='number_of_data', y='field_name', data=df_split, palette="Spectral_r")
        for index, data in enumerate(df_split.number_of_data):
            plot.text(x=data+500, y=index, s=data, color='black', ha='left', va='center')
        fig.savefig(join(output_path, 'whole_{}_{}.png'.format(num_name, TYPENAME)), bbox_inches='tight')

        fig1 = plt.figure(figsize=(20, 15))
        plot1 = sns.barplot(x='number_of_data', y='field_name', data=df_train, palette="Spectral_r")
        for index, data in enumerate(df_train.number_of_data):
            plot1.text(x=data+500, y=index, s=data, color='black', ha='left', va='center')
        fig1.savefig(join(output_path, 'train_{}_{}.png'.format(num_name, TYPENAME)), bbox_inches='tight')

        fig2 = plt.figure(figsize=(20, 15))
        plot2 = sns.barplot(x='number_of_data', y='field_name', data=df_test, palette="Spectral_r")
        for index, data in enumerate(df_test.number_of_data):
            plot2.text(x=data+500, y=index, s=data, color='black', ha='left', va='center')
        fig2.savefig(join(output_path, 'test_{}_{}.png'.format(num_name, TYPENAME)), bbox_inches='tight')

        #fig2 = plt.figure(figsize=(20, 15))
        #sns.barplot(x='field_name', y='number_of_data', data=df_test, palette="Spectral_r")
        #plt.xticks(rotation=90)
        #fig2.savefig(join(output_path, 'number_of_data_test_{}.png'.format(TYPENAME)), bbox_inches='tight')


    if 'violin3' in mode:

        print("violin plot")
        #cv_result_sherlock = pd.read_csv(join(result_path, 'sherlock', TYPENAME, s_model_name, 'cv_results.csv'))
        #cv_result_topic1 = pd.read_csv(join(result_path, 'sherlock', TYPENAME, st_model_name1, 'cv_results.csv'))
        #cv_result_topic2 = pd.read_csv(join(result_path, 'sherlock', TYPENAME, st_model_name2, 'cv_results.csv'))
        cv_result_crf_sherlock = pd.read_csv(join(result_path, 'CRF', TYPENAME, crfs_model_name, 'cv_results.csv'))
        cv_result_crf_topic1 = pd.read_csv(join(result_path, 'CRF', TYPENAME, crft_model_name1, 'cv_results.csv'))
        cv_result_crf_topic2 = pd.read_csv(join(result_path, 'CRF', TYPENAME, crft_model_name2, 'cv_results.csv'))

        cv_result_dict = data_for_violinplot3(cv_result_crf_sherlock, cv_result_crf_topic1, topic1_num, cv_result_crf_topic2, topic2_num)
        cv_result = pd.DataFrame(cv_result_dict)


        fig = plt.figure(figsize=(15, 10))
        sns.violinplot(x='acc_type', y='accuracy', data=cv_result, hue='model', palette='Spectral_r')
        plt.legend(loc='lower right')
        plt.ylim([0, 1])
        fig.savefig(join(output_path, 'violin_{}_{}.png'.format(violinplot_name, TYPENAME)), bbox_inches = 'tight')

    if 'violin2' in mode:

        print("violin plot")
        cv_result_sherlock = pd.read_csv(join(result_path, 'sherlock', TYPENAME, s_model_name, 'cv_results.csv'))
        #cv_result_topic = pd.read_csv(join(result_path, 'sherlock', TYPENAME, st_model_name1, 'cv_results.csv'))
        cv_result_crf_sherlock = pd.read_csv(join(result_path, 'CRF', TYPENAME, crfs_model_name, 'cv_results.csv'))
        #cv_result_crf_topic = pd.read_csv(join(result_path, 'CRF', TYPENAME, crft_model_name1, 'cv_results.csv'))

        cv_result_dict = data_for_violinplot(cv_result_sherlock, cv_result_crf_sherlock)
        cv_result = pd.DataFrame(cv_result_dict)

        fig = plt.figure(figsize=(10, 15))
        sns.violinplot(x='acc_type', y='accuracy', data=cv_result, hue='model', palette='Spectral_r')
        plt.legend(loc='lower right')
        plt.ylim([0, 1])
        fig.savefig(join(output_path, 'violin_{}_{}.png'.format(violinplot_name, TYPENAME)), bbox_inches = 'tight')

    if 'bar' in mode:
        print("bar plot")
        #sherlock = data_for_barplot('sherlock', s_model_name)
        #topic1 = data_for_barplot('sherlock', st_model_name1)
        topic2 = data_for_barplot('sherlock', st_model_name2)
        print(topic2)
        #crf_sherlock = data_for_barplot('CRF', crfs_model_name)
        #crf_topic1 = data_for_barplot('CRF', crft_model_name1)
        crf_topic2 = data_for_barplot('CRF', crft_model_name2)
        print(crf_topic2)

        '''
        fig1 = plt.figure(figsize=(20, 5))
        sns.barplot(x='types', y='accuracy', data=crf_sherlock, palette="Spectral_r")
        plt.xticks(rotation=90)
        fig1.savefig(join(output_path, 'sherlock_{}_{}.png'.format(barplot_name, TYPENAME)), bbox_inches='tight')

        fig2 = plt.figure(figsize=(20, 5))
        sns.barplot(x='types', y='accuracy', data=crf_topic1, palette="Spectral_r")
        plt.xticks(rotation=90)
        fig2.savefig(join(output_path, 'topic1_{}_{}.png'.format(barplot_name, TYPENAME)), bbox_inches='tight')

        fig3 = plt.figure(figsize=(20, 5))
        sns.barplot(x='types', y='accuracy', data=crf_topic2, palette="Spectral_r")
        plt.xticks(rotation=90)
        fig3.savefig(join(output_path, 'topic2_{}_{}.png'.format(barplot_name, TYPENAME)), bbox_inches='tight')
        '''

        #sherlock = crf_sherlock.rename(columns={"accuracy": "sherlock"})
        #topic1 = crf_topic1.rename(columns={"accuracy": "topic{}".format(topic1_num)})
        #topic2 = crf_topic2.rename(columns={"accuracy": "topic{}".format(topic2_num)})

        #final = sherlock[['sherlock', 'types']].join(topic1[['topic{}'.format(topic1_num), 'types']].set_index('types'), on='types')
        #final = final[['sherlock', 'topic{}'.format(topic1_num), 'types']].join(topic2[['topic{}'.format(topic2_num), 'types']].set_index('types'), on='types')

        topic2 = topic2.rename(columns={"avg_acc": "single_col"})
        crf_topic2 = crf_topic2.rename(columns={"avg_acc": "CRF"})

        final = topic2[['single_col', 'types']].join(crf_topic2[['CRF', 'types']].set_index('types'), on='types')
        final = final.sort_values(["CRF", "single_col"], ascending=(False, False))

        final = pd.melt(final, id_vars="types", var_name="mode", value_name="accuracy")
        final = final.sort_values(by=['accuracy'], ascending=False)

        #final = final.sort_values(by=['CRF'], ascending=False)
        final.to_csv(join(output_path, '{}_{}.csv'.format(barplot_name, TYPENAME)))
        fig = plt.figure(figsize=(20, 5))
        sns.barplot(x='types', y='accuracy', data=final, hue='mode', palette="Spectral_r")
        plt.xticks(rotation=90)
        fig.savefig(join(output_path, 'accuracy_per_types_{}_{}.png'.format(barplot_name, TYPENAME)), bbox_inches = 'tight')

