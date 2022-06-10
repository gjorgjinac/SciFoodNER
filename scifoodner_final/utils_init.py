import os
import csv
import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split


def make_dirs(dataset, fold_count):
    if not os.path.isdir(f'folds_{fold_count}/{dataset}'):
        os.makedirs(f'folds_{fold_count}/{dataset}')
    if not os.path.isdir(f'results_{fold_count}/{dataset}'):
        os.makedirs(f'results_{fold_count}/{dataset}')


def assign_sentence_indices(df):
    # assign sentence indices
    sentence_index = 0
    for index, row in df.iterrows():
        df.loc[index, 'sentence_index'] = sentence_index
        if row['word'] == '.' and not (
                index > 1 and index + 1 < df.shape[0] and df.loc[index + 1, 'word'].isnumeric() and df.loc[
            index - 1, 'word'].isnumeric()):
            sentence_index += 1

    # assign full sentence column
    sentences = df.groupby('sentence_index')['word'].transform(lambda x: ' '.join([str(xx) for xx in x]))
    unique_sentences = sentences.drop_duplicates()
    df['full sentence'] = sentences.values
    return df, unique_sentences


def generate_new_folds(df, dataset, unique_sentences, fold_count):
    for i in range(0, fold_count):
        if os.path.isfile(f'folds_{fold_count}/{dataset}/train_{i}'):
            print(
                'Training file exists for at least one fold. Please remove existing folds if you wish to proceed with creating new ones')
            return

    kf = KFold(n_splits=fold_count)

    for fold, (train_index, test_index) in enumerate(kf.split(unique_sentences)):
        train_index, val_index = train_test_split(train_index, test_size=0.1)

        train_sentences = df[df['sentence_id'].isin(train_index)]
        test_sentences = df[df['sentence_id'].isin(test_index)]
        val_sentences = df[df['sentence_id'].isin(val_index)]
        [split.to_csv(f'folds_{fold_count}/{dataset}/{split_name}_{fold}') for split_name, split in
         [('train', train_sentences), ('val', val_sentences), ('test', test_sentences)]]

        
def read_df(dataset):
    if dataset=='cafeteria':
        df = pd.read_csv('NER_data/cafeteria.tsv', sep='\t').dropna()
    else:
        df = pd.read_csv(f'NER_data/{dataset}.csv', index_col=[0])

    df.columns=['word','tag']
    df['word']=df['word'].astype(str)
    df['tag']=df['tag'].astype(str)
    return df
    
def init(dataset, fold_count):
    df=read_df(dataset)
    make_dirs(dataset, fold_count)
    df, unique_sentences = assign_sentence_indices(df)
    dataset_labels = list(df['tag'].drop_duplicates().values)
    unique_sentences=unique_sentences.reset_index(drop=True)
    df=df.rename(columns={"sentence_index":"sentence_id", "word":"words", "tag":"labels"})
    return df, unique_sentences, dataset_labels
