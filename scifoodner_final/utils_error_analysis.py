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

def parse_predictions(dataset, test, model_name, fold, result_dir):
    predictions = pd.read_csv(f'{result_dir}/{dataset}/{model_name}_full_{fold}_predictions.csv', index_col=0)
    predictions['sentence_id'] = test['sentence_id'].drop_duplicates().values
    predictions = predictions.melt(id_vars=['sentence_id'], 
            var_name="word_id", 
            value_name="labels_pred")
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

def parse_test_values(dataset, fold, fold_dir):
    df = pd.read_csv(f'{fold_dir}/{dataset}/test_{fold}', index_col=0)
    df['sentence_id'] = df['sentence_id'].astype(int)
    df['word_id'] = word_id_gen(df['sentence_id'].to_list())
    return df

def get_merged(dataset, model_name, fold, result_dir, fold_dir):
    test = parse_test_values(dataset, fold,fold_dir)
    predictions = parse_predictions(dataset, test, model_name, fold, result_dir)
    merged = predictions.merge(test, on=['word_id','sentence_id'], how='inner')
    return merged.dropna(), test, predictions

@log
def get_merged_all(datasets, model_names, folds, result_dir, fold_dir):
    frames = []
    for dataset in datasets:
        for fold in folds:
            for model_name in model_names:
                merged_df, _, _ = get_merged(dataset, model_name, fold, result_dir,fold_dir)
                merged_df['fold'] = fold
                merged_df['model_name'] = model_name
                merged_df['dataset']=dataset
                frames.append(merged_df)
    df = pd.concat(frames)
    return df






