import pandas as pd
import os
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import json
import configargparse

TYPENAME = os.environ['TYPENAME']
header_path = join(os.environ['BASEPATH'], 'extract/out/headers')

tmp_path = 'out/data_list'
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

p = configargparse.ArgParser()
# p.add('-c', '--config_file', is_config_file=True, help='config file path')
p.add('--header_file', type=str, help='name of extracted header file')
p.add('--output', type=str, help='name of output file')
p.add('--multi_col_only', type=bool, default=False, help='filtering only the tables with multiple columns')

args = p.parse_args()
header_file_name = args.header_file
output_file = args.output
multi_col = args.multi_col_only

#print('Spliting {}'.format(header_file_name))
#multi_tag = '_multi-col' if multi_col else ''
    
split_dic = {}

header_file = join(header_path, header_file_name)
df = pd.read_pickle(header_file)

#if multi_col:
 #   df.loc[:, 'col_count'] = df.apply(lambda x: len(eval(x['field_names'])), axis=1)
 #   df= df[df['col_count']>1]

train_list, test_list= train_test_split(df['dataset_id'], test_size=0.2, random_state=42)
train_list, val_list= train_test_split(train_list, test_size=0.2, random_state=42)
split_dic = {'train':list(train_list), 'val':list(val_list), 'test':list(test_list)}

total_size = len(split_dic['train'])
sample_from = split_dic['train']

for s_p in sorted(sample_percentages, reverse=True):

    sample_size = int(s_p/100*total_size)
        
    new_sample = np.random.choice(sample_from, sample_size, replace=False)
    sample_from = new_sample
    split_dic['train_{}per'.format(s_p)] = list(new_sample)

print(split_dic.keys())

print("Done, {} training tables{}, {} validation tables{}, {} testing tables{}".format(len(split_dic['train']), multi_tag,
                                                               len(split_dic['val']), multi_tag,
                                                               len(split_dic['test']), multi_tag))

with open(join(tmp_path, '{}_{}{}.json'.format(output_file, TYPENAME, multi_tag)),'w') as f:
    json.dump(split_dic, f)