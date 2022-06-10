import pandas as pd
from collections import namedtuple
from sklearn.metrics import classification_report
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt

def log(f):
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        res = f(*args, **kwargs)
        toc = dt.datetime.now()
        if hasattr(res, 'shape'):
            print(f'{f.__name__} took={toc-tic} shape={res.shape}')
        else:
            print(f'{f.__name__} took={toc-tic}')
        return res
    return wrapper

def parse_predictions(dataset, test, model_name, fold, result_dir, split):
    predictions = pd.read_csv(f'{result_dir}/{dataset}/{model_name}_full_{split}_{fold}_predictions.csv', index_col=0)
    predictions['sentence_id'] = test['sentence_id'].drop_duplicates().values
    #predictions = predictions.melt(id_vars=['sentence_id'],  var_name="word_id", value_name="labels_pred")
    predictions['word_id'] = predictions['word_id'].astype(int)
    return predictions


def word_id_gen(sentence_ids):
    current_val = None
    current_index = -1
    new_l = []
    for x in sentence_ids:
        if x != current_val:
            current_index = -1
            current_val = x
        current_index+=1
        new_l.append(current_index)
    return new_l

@log
def parse_test_values(dataset, fold, fold_dir,split):
    df = pd.read_csv(f'{fold_dir}/{dataset}/{split}_{fold}', index_col=0)
    df['sentence_id'] = df['sentence_id'].astype(int)
    df['word_id'] = word_id_gen(df['sentence_id'].to_list())
    return df


def sentence_id_gen(df, start_id):
    sentence_id=start_id
    for index, row in df.iterrows():
        df.loc[index, 'sentence_id']=sentence_id
        if row['words']=='.':
            sentence_id+=1
    return df


def parse_predictions(dataset, model_name, fold, result_dir, split, test):
    predictions = pd.read_csv(f'{result_dir}/{dataset}/{model_name}_full_{split}_{fold}_predictions.csv', index_col=0)
    predictions=sentence_id_gen(predictions, test['sentence_id'].values[0])
    predictions['word_id'] = word_id_gen(predictions['sentence_id'].astype(int).to_list())
    return predictions

@log
def get_merged_all(datasets, model_names, folds, result_dir, fold_dir, split):
    
    merged_all=pd.DataFrame()
    for dataset in datasets:
        for fold in folds:
            test=parse_test_values(dataset, fold, fold_dir,split)
            for model_name in model_names:
                predictions=parse_predictions(dataset, model_name, fold, result_dir, split, test)
                merged = predictions.merge(test, on=['word_id','sentence_id', 'words'], how='inner')
                merged['fold']=fold
                merged['model_name']=model_name
                merged['dataset']=dataset
                merged_all=merged_all.append(merged)
    return merged_all






