import pandas as pd
from utils_init import *
from utils_train import *


def run(dataset, fold_count=5, device_to_use=0, 
        bert_model_name='bert',
        run_k_fold=False,
        run_generate_new_folds=False):
    df, unique_sentences, dataset_labels = init(dataset, fold_count)
    fold_count=int(fold_count)
    if run_generate_new_folds:
        generate_new_folds(df, dataset, unique_sentences, fold_count)
    if run_k_fold:
        k_fold(dataset, fold_count, dataset_labels, device_to_use, bert_model_name)
