import os
import argparse

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import concatenate_datasets, load_dataset
from datasets import Dataset, DatasetDict
from sklearn.metrics import recall_score, precision_score

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
import torch

# import sys
# # setting path
# sys.path.append('../notebook')

from utils.KeyPointEvaluator import *
from utils.KeyPointEvaluatorRev import *

torch.cuda.empty_cache()


def get_model(model_dir, category_name):
    model_path = f'{model_dir}/{category_name}'
    latest_checkpoint = os.listdir(model_path)[-1]
    model_path = f'{model_path}/{latest_checkpoint}/'

    return SentenceTransformer(model_path)


def evaluate_model(df, model, in_category=True):
    comment_df, kp_df, labels_df = prepare_comment_kp_label_input(df)

    # Perform prediction on the validation/test dataframes
    preds = perform_preds(model, comment_df, kp_df, in_category)

    # Get the best predicted KP for every review sentence
    merged_df = get_predictions(preds, labels_df, comment_df)
    merged_df = merged_df.drop_duplicates(subset=['comment_id', 'key_point_id'])

    precisions = calc_mean_average_precision(merged_df, "label")

    return merged_df, precisions


def do_eval(base_model_dir, experiment_type, test_data_path, data_configuration):
    df = pd.read_pickle(test_data_path)

    perf = []
    for category_name in sorted(df['topic'].unique()):
        model_dir = f"{base_model_dir}/{experiment_type}/"
        model = get_model(model_dir, category_name)

        if data_configuration == 'all_comments':
            merged_df, precisions = evaluate_model(df[df['topic'] == category_name], model)
        else:
            merged_df, precisions = evaluate_model(df[(df['topic'] == category_name)
                                                      & (df['isMultiAspect'] == True)], model)

        del model
        torch.cuda.empty_cache()

        perf += [pd.Series({'Business Category': category_name, 'Average Precision': precisions})]

    perf_df = pd.concat(perf, axis=1).T
    print(f"########## {experiment_type.upper()} ({'All comments' if data_configuration == 'all_comments' else 'Multi-opinion comments'}) EVALUATION ##########")
    print(perf_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, default="./checkpoints/ABKPA",
                        help="Path to the base directory of model checkpoints of ABKPA.")
    parser.add_argument("--experiment_type", type=str, default=None,
                        help="The experiment type. Must either be 'in_category' or 'out_of_category'.")
    parser.add_argument("--data_configuration", type=str, default=None,
                        help="The experiment data configuration. Must either be 'all_comments' or 'multi_opinion_comments'.")
    parser.add_argument("--test_data_path", type=str, default='./data/Evaluation/test_data.pkl',
                        help="The path to the pkl file of the test data.")

    args = parser.parse_args()

    experiment_type = args.experiment_type.replace("_", '-')
    data_configuration = args.data_configuration

    if experiment_type in ["in-category", 'out-of-category']:
        if data_configuration in ["all_comments", 'multi_opinion_comments']:
            do_eval(args.base_model_dir, experiment_type, args.test_data_path, data_configuration)
        else:
            print("Invalid data configuration. The data configuration must either be 'all_comments' or 'multi_opinion_comments'.")
    else:
        print("Invalid experiment type. The experiment type must either be 'in_category' or 'out_of_category'.")
