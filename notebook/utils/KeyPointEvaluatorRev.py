import pandas as pd
import json
from .KeyPointEvaluator import *

def match_comment_with_keypoints(result, kp_dict, comment_dict):
    for comment, comment_embedding in comment_dict.items():
        result[comment] = {}
        for kp, kp_embedding in kp_dict.items():
            result[comment][kp] = util.pytorch_cos_sim(comment_embedding, kp_embedding).item()
        
        #Applying softmax
        kp_scores = list(result[comment].items())
        kp_ids, kp_scores = zip(*kp_scores)
        result[comment] = {kp_id:score for kp_id, score in zip(kp_ids, kp_scores)}
        

    return result

def predict(model, comment_df, keypoint_df, output_path, append_topic=False):
    comment_keypoints = {}
    for topic in comment_df.topic.unique():
        for stance in [-1, 1]:
            topic_keypoints_ids = keypoint_df[(keypoint_df.topic==topic) & (keypoint_df.stance==stance)]['key_point_id'].tolist()
            topic_keypoints = keypoint_df[(keypoint_df.topic==topic) & (keypoint_df.stance==stance)]['key_point'].tolist()
            if append_topic:
                topic_keypoints = [topic + ' <SEP> ' + x for x in topic_keypoints]
                
            topic_keypoints_embeddings = model.encode(topic_keypoints)
            topic_kp_embed = dict(zip(topic_keypoints_ids, topic_keypoints_embeddings))

            topic_comments_ids = comment_df[(comment_df.topic==topic) & (comment_df.stance==stance)]['comment_id'].tolist()
            topic_comments = comment_df[(comment_df.topic==topic) & (comment_df.stance==stance)]['comment'].tolist()
            topic_comments_embeddings = model.encode(topic_comments)
            topic_comment_embed= dict(zip(topic_comments_ids, topic_comments_embeddings))

            comment_keypoints = match_comment_with_keypoints(comment_keypoints, topic_kp_embed, topic_comment_embed)
    
    json.dump(comment_keypoints, open(output_path, 'w'))
    
    return comment_keypoints

def get_predictions(preds, labels_df, comment_df):
    comment_df = comment_df[["comment_id", "comment_id_sent", "topic"]]
    predictions_df = load_predictions(preds)
    #make sure each comment_id has a prediction
    predictions_df = pd.merge(comment_df, predictions_df, how="left", on="comment_id")
    predictions_df = predictions_df.rename(columns={'comment_id': 'comment_id_new', 'comment_id_sent': 'comment_id'})

    #handle comments with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    #merge predicted comment-KP pair with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["comment_id", "key_point_id"])

    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    
    return merged_df

def prepare_comment_kp_label_input(df):
    comment_df = df[['comment_id', 'topic', 'comment', 'full_comment', 'isMultiAspect']]\
        .drop_duplicates(subset=['comment_id']).reset_index(drop=True)
    comment_df = comment_df.explode(['comment'])
    comment_df = comment_df.groupby(['comment_id']).apply(lambda x: x.reset_index(drop=True).reset_index()).reset_index(drop=True)
    comment_df = comment_df.rename(columns={'comment_id': 'comment_id_sent'})
    comment_df['comment_id'] = comment_df['comment_id_sent'] + "_" + comment_df['index'].astype(str)
    
    kp_df = df[['key_point_id', 'topic', 'key_point', 'full_key_point']].drop_duplicates(subset=['key_point_id']).reset_index(drop=True)        
    
    labels_df = df[['comment_id', 'key_point_id', 'label']]
    
    return comment_df, kp_df, labels_df