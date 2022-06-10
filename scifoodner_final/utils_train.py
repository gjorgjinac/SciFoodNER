import os
import random

import numpy as np
import pandas as pd
import torch
from simpletransformers.ner import NERModel, NERArgs
import shutil
from sklearn.metrics import classification_report

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

bert_model_config={'biobert':'dmis-lab/biobert-v1.1', 'roberta': 'roberta-base', 'bert':'bert-base-uncased', 'scibert':'allenai/scibert_scivocab_uncased'}

def evaluate_model(train, val, test, fold_count, result_file_name, dataset, dataset_labels, device_to_use, bert_model_name, fold_number):
    result_file=f'results_{fold_count}/{dataset}/{bert_model_name}_full_test_{fold_number}.csv'
    if os.path.isdir(f'outputs_{fold_count}/{dataset}_{fold_number}_{result_file_name}'):
        shutil.rmtree(f'outputs_{fold_count}/{dataset}_{fold_number}_{result_file_name}', ignore_errors=True)

    print(result_file)
    if os.path.isfile(result_file):
        print(f'Result file exists: {result_file}')
        return False
    ner_model_args = NERArgs(num_train_epochs=100, save_model_every_epoch=False, overwrite_output_dir=True,
                             output_dir=f'outputs_{fold_count}/{dataset}_{fold_number}_{result_file_name}',
                             best_model_dir=f'outputs_{fold_count}/{dataset}_{fold_number}_{result_file_name}/best',
                             save_eval_checkpoints=False, save_steps=-1, max_seq_length=200,
                             evaluate_during_training_verbose=True, evaluate_during_training=True,
                             early_stopping_consider_epochs=True, use_early_stopping=True, early_stopping_patience=5,
                             early_stopping_delta=5e-3)

    ner_model = NERModel(
        bert_model_name if bert_model_name=='roberta' else 'bert',
        bert_model_config[bert_model_name],
        args=ner_model_args,
        use_cuda=True,
        labels=dataset_labels,
        cuda_device=device_to_use
    )
    ner_model.train_model(train, eval_data=val)
    save_predictions(ner_model=ner_model, test=test, split='test', fold_count=fold_count, bert_model_name=bert_model_name, dataset=dataset, fold_number=fold_number)
    save_predictions(ner_model=ner_model, test=train, split='train', fold_count=fold_count, bert_model_name=bert_model_name, dataset=dataset, fold_number=fold_number)
    return ner_model


def k_fold(dataset, fold_count, dataset_labels, device_to_use, bert_model_name):
    for fold in range(0, fold_count):
        print(f'Fold number: {fold}')
        train_sentences, val_sentences, test_sentences = [
            pd.read_csv(f'folds_{fold_count}/{dataset}/{split_name}_{fold}', index_col=[0]).dropna().astype(str) for split_name in
            ['train', 'val', 'test']]

        evaluate_model(train_sentences, val_sentences, test_sentences, fold_count, f'full_{fold}', dataset, dataset_labels,
                       device_to_use, bert_model_name, fold)


def save_predictions(ner_model, test, fold_count, bert_model_name, dataset, fold_number, split):
    results_file = f'results_{fold_count}/{dataset}/{bert_model_name}_full_{split}_{fold_number}.csv'
    predictions_file = f'results_{fold_count}/{dataset}/{bert_model_name}_full_{split}_{fold_number}_predictions.csv'
    test_model_inputs = [list(test[test['sentence_id']==sentence_id]['words'].astype(str).values) for sentence_id in test['sentence_id'].drop_duplicates()]
    preds, model_outputs=ner_model.predict(test_model_inputs, split_on_space=False)
    print(preds)
    predictions=pd.DataFrame([ list(p.items())[0] for sentence_preds in preds for p in sentence_preds])
    print(predictions)
    predictions.columns=['words','predictions']
    predictions.to_csv(predictions_file)
    test['predictions']=predictions['predictions']
    all_test_samples_size=test.shape[0]
    test=test.dropna().astype(str)
    print(f'Dropped nan predictions: {all_test_samples_size - test.shape[0]}')
    report = classification_report(test['predictions'],test['labels'], output_dict=True)
    report = pd.DataFrame(report)
    print(report)
    report.to_csv(results_file)
                                   
