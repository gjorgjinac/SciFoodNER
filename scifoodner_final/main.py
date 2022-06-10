import argparse
from utils_run import run
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--bert_model_name', required=False, default='bert')
parser.add_argument('--fold_count', default=5, required=False)
parser.add_argument('--device_to_use', default=0, required=False)
parser.add_argument('--run_k_fold', default=False, required=False)
parser.add_argument('--run_generate_new_folds', default=False, required=False)


args = parser.parse_args()
print(args.__dict__)
run(**args.__dict__)