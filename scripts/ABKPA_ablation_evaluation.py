import os
import argparse

import json
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from datasets import concatenate_datasets, load_dataset
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from sklearn.metrics import recall_score, precision_score
from utils.track_1_kp_matching import *

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
import torch
import spacy
import nltk

# import sys
# # setting path
# sys.path.append('../notebook')

from utils.KeyPointEvaluator import *
from utils.KeyPointEvaluatorRev import *


torch.cuda.empty_cache()
nlp = spacy.load('en_core_web_md')


def prepare_comment_kp_label_input(df):
    df['num_of_aspects'] = df['comment'].str.len()
    df = df.rename(columns={'comment_id': 'comment_id_sent'})
    df['comment_id'] = df['comment_id_sent'] + "_" + df['num_of_aspects'].astype(str)

    comment_df = df[['comment_id', 'topic', 'full_comment', 'num_of_aspects',
                     'aspects_x', 'opinions_x', 'opinion_aspect_pairs_x', 'sentiments_x', 'attributes']]
    comment_df = comment_df.drop_duplicates(subset=['comment_id']).reset_index(drop=True)
    comment_df.columns = [col.replace("_x", "") for col in comment_df.columns]

    kp_df = df[['key_point_id', 'topic', 'full_key_point',
                'aspects_y', 'opinions_y', 'opinion_aspect_pairs_y', 'sentiments_y', 'attributes']]
    kp_df = kp_df.drop_duplicates(subset=['key_point_id']).reset_index(drop=True)
    kp_df = kp_df.rename(columns={'full_key_point': 'key_point'})
    kp_df.columns = [col.replace("_y", "") for col in kp_df.columns]
    kp_df['opinion_aspect_pairs'] = kp_df['opinion_aspect_pairs'].apply(lambda x: ast.literal_eval(x))
    kp_df['sentiments'] = kp_df['sentiments'].apply(lambda x: ast.literal_eval(x))

    labels_df = df[['comment_id', 'key_point_id', 'label']]

    return comment_df, kp_df, labels_df


def comment_kp_absa_match(comment_df, kp_df):
    comment_df = comment_df.explode(['aspects', 'opinions', 'opinion_aspect_pairs', 'sentiments', 'attributes'])
    kp_df = kp_df.explode(['opinion_aspect_pairs', 'sentiments'])

    # Matching based on ABSA predictions of comment and KPs
    sent_kp_df = comment_df.astype(str).merge(kp_df.astype(str), how='left', on=['topic']).dropna(subset=['key_point'])
    sent_kp_df = sent_kp_df.reset_index(drop=True)
    # Label the matching/non-matching pairs
    sent_kp_df.loc[
        (sent_kp_df['aspects_x'] == sent_kp_df['aspects_y']) & (
                    sent_kp_df['sentiments_x'] == sent_kp_df['sentiments_y']),
        'score'
    ] = 1
    sent_kp_df.loc[
        (sent_kp_df['sentiments_x'] != sent_kp_df['sentiments_y']),
        'score'
    ] = 0
    sent_kp_df = implicit_aspects_matching(sent_kp_df)

    predictions_df = sent_kp_df.groupby(['topic', 'comment_id', 'opinion_aspect_pairs_x']).apply(
        select_best_kp_per_opinion).reset_index(drop=True)
    predictions_df = predictions_df.drop_duplicates(subset=['comment_id', 'key_point_id'])

    return predictions_df


def select_best_kp_per_opinion(grp):
    best_match = grp.sort_values(by=['score'], ascending=False)
    best_match = best_match.iloc[0]
    return best_match


def implicit_aspects_matching(sent_kp_df):
    sent_kp_df['predicted_by_cosine'] = False

    import warnings
    warnings.filterwarnings("error")
    implicit_match_analyze_mask = pd.isnull(sent_kp_df['score'])
    sent_kp_df[implicit_match_analyze_mask] = sent_kp_df[implicit_match_analyze_mask].apply(
        calculate_aspects_semantic_similarity, axis=1)
    warnings.filterwarnings("ignore")

    return sent_kp_df


unfound_tokens = []
def calculate_aspects_semantic_similarity(row):
    tokens = nlp(row['aspects_x'] + " | " + row['aspects_y'])
    sep_index = [token.i for token in tokens if token.text == '|'][0]
    token1, token2 = tokens[:sep_index], tokens[sep_index + 1:]
    row['aspects_x_len'] = len(token1)
    row['aspects_y_len'] = len(token2)
    #     print(token1, token2)
    try:
        row['score'] = token2.similarity(token1)
    except:
        #         display(row)
        #         unfound_tokens += [token1]
        row['score'] = 0
    row['predicted_by_cosine'] = True

    return row


def label_implicit_matching_pair(row):
    token1 = row['aspects_x'][0]
    token2 = row['aspects_y'][0]

    # Best
    if row['aspects_x_len'] >= row['aspects_y_len']:
        if token2 in token1 and len(token1) > len(token2) and row['aspects_x_len'] > row[
            'aspects_y_len']:  # e.g. mexican food vs food
            row['label'] = 1
    return row


def evaluate(df):
    comment_df, kp_df, labels_df = prepare_comment_kp_label_input(df)
    predictions_df = comment_kp_absa_match(comment_df, kp_df)

    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["comment_id", "key_point_id"])
    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)

    precisions = [(t, get_ap(group, "label_strict")) for t, group in merged_df.groupby(["topic"])]

    return merged_df, precisions


def do_eval(test_data_path, data_configuration):
    df = pd.read_pickle(test_data_path)

    if data_configuration == "all_comments":
        merged_df, precisions = evaluate(df)
    else:
        merged_df, precisions = evaluate(df[df['isMultiAspect'] == True])

    perf = []
    for (category_name, precision) in precisions:
        perf += [pd.Series({'Business Category': category_name, 'Average Precision': precision})]
    perf_df = pd.concat(perf, axis=1).T

    print(f"########## ({'All comments' if data_configuration == 'all_comments' else 'Multi-opinion comments'}) EVALUATION ##########")
    print(perf_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, default="./checkpoints/ABKPA",
                        help="Path to the base directory of model checkpoints of ABKPA.")
    parser.add_argument("--data_configuration", type=str, default=None,
                        help="The experiment data configuration. Must either be 'all_comments' or 'multi_opinion_comments'.")
    parser.add_argument("--test_data_path", type=str, default='./data/Evaluation/test_data.pkl',
                        help="The path to the pkl file of the test data.")

    args = parser.parse_args()

    data_configuration = args.data_configuration

    if data_configuration in ["all_comments", 'multi_opinion_comments']:
        do_eval(args.test_data_path, data_configuration)
    else:
        print("Invalid data configuration. The data configuration must either be 'all_comments' or 'multi_opinion_comments'.")