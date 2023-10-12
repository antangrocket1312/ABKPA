# Functions to label id
i = 0
curr_topic = ""
def label_group_id(grp_df):
    global i
    grp_df['group_id'] = i
    grp_df = grp_df.reset_index(drop=True)
    grp_df = grp_df.reset_index()
    i += 1
    return grp_df

def label_kp_id(grp_df):
    global i
    global curr_topic
    if grp_df['topic'].iloc[0] != curr_topic:
        i = 0
        curr_topic = grp_df['topic'].iloc[0]
    grp_df['kp_id'] = i
    grp_df = grp_df.reset_index(drop=True)
    i += 1
    return grp_df

def label_arg_id(grp_df):
    global i
    global curr_topic
    if grp_df['topic'].iloc[0] != curr_topic:
        i = 0
        curr_topic = grp_df['topic'].iloc[0]
    grp_df['comm_id'] = i
    grp_df = grp_df.reset_index(drop=True)
    i += 1
    return grp_df


def create_arg_kp_label_df(datasets, subset):
  small_test_dataset = datasets[subset].shuffle(seed=42)
  small_test_dataset.set_format("pandas")
  small_df = small_test_dataset[:]
  small_df = small_df.rename(columns={'text': 'comment'})

  global i
  i = 0
  small_df = small_df.groupby(['topic']).apply(label_group_id).reset_index(drop=True)

  i = 0
  small_df = small_df.groupby(['topic', 'comment']).apply(label_arg_id).reset_index(drop=True)

  i = 0
  small_df = small_df.groupby(['topic', 'key_point']).apply(label_kp_id).reset_index(drop=True)

  small_df['comment_id'] = "arg_" + small_df['group_id'].astype(str) + "_" + small_df['comm_id'].astype(str)
  small_df['key_point_id'] = "kp_" + small_df['group_id'].astype(str) + "_" + small_df['kp_id'].astype(str)

  arguments_df = small_df.drop_duplicates(subset=['group_id', 'comm_id']).reset_index(drop=True) \
    [['comment_id', 'comment', 'full_comment', 'topic', 'domain', 'predicted_WA_x']]

  keypoints_df = small_df.drop_duplicates(subset=['group_id', 'kp_id']).reset_index(drop=True) \
    [['key_point_id', 'key_point', 'full_key_point', 'topic', 'domain', 'predicted_WA_y']]

  labels_df = small_df[:].sort_values(by=['group_id', 'index'])[['comment_id', 'key_point_id', 'label']]
  labels_df['label'] = labels_df.label.apply(lambda x: int(x))

  arguments_df = arguments_df.rename(columns={'predicted_WA_x': 'predicted_WA'})
  keypoints_df = keypoints_df.rename(columns={'predicted_WA_y': 'predicted_WA'})

  return arguments_df.reset_index(drop=True), keypoints_df.reset_index(drop=True), labels_df.reset_index(drop=True)