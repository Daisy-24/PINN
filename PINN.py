import numpy as np
import pandas as pd
import pickle
from itertools import chain
from scipy.sparse import lil_matrix
import copy
import cupy as cp
import similaripy as sim


# Read data
data_train = pd.read_csv('data/train_baskets_all.csv')
data_test = pd.read_csv('data/test_baskets.csv')

# Change float to int
data_train['StockCode'] = data_train['StockCode'].astype(int)
data_train['InvoiceNo'] = data_train['InvoiceNo'].astype(int)
data_train['CustomerID'] = data_train['CustomerID'].astype(int)
data_train['transaction_number'] = data_train['transaction_number'].astype(int)

data_test['StockCode'] = data_test['StockCode'].astype(int)
data_test['InvoiceNo'] = data_test['InvoiceNo'].astype(int)
data_test['CustomerID'] = data_test['CustomerID'].astype(int)
data_test['transaction_number'] = data_test['transaction_number'].astype(int)

num_nearest_neighbors = 250
group_decay_rate = -1.4
history_rate = 0.1
alpha = 0.2  # Personal component
topk = 50
item_size = max(data_train['StockCode'].max(),data_test['StockCode'].max())
user_size = data_train['CustomerID'].nunique()
test_number = data_test.shape[0]
train_users = np.unique(data_train['CustomerID'])
train_items = np.unique(data_train['StockCode'])
item_list = data_test['StockCode'].unique().tolist()


df_train_item = data_train.groupby(['StockCode'])
df_test_user = data_test.groupby(['CustomerID'])
# Create a dictionary for [Item: User]
data_dict = {}
for index, content in df_train_item:
    item_id = content['StockCode'].values[0]
    user_id = content['CustomerID'].unique().tolist()
    if item_id in item_list:
        data_dict[item_id] = user_id

# Create a dictionary for [User: itme]
user_to_item_test_dict= {}
for index, content in df_test_user:
    item_id = content['StockCode'].values
    user_id = content['CustomerID'].values[0]
    user_to_item_test_dict[user_id] = item_id

# ----------------History component----------------------
user_item_matrix = np.zeros((user_size, item_size),dtype=np.float16)

for i, user in enumerate(train_users):
    user_data = data_train[data_train['CustomerID'] == user]
    project_set = user_data['StockCode'].unique()
    trans_num = user_data['InvoiceNo'].unique()

    for project in project_set:
        project_trans = user_data[user_data['StockCode'] == project]['transaction_number'].values
        for idx, order in enumerate(project_trans):
            score = ((len(project_trans) - idx) ** group_decay_rate)
            user_item_matrix[i, project - 1] += score

# ----------------Personal and neighbor component----------------------
data_chunk = [{} for _ in range(user_size)]

for product, users in data_dict.items():
    for user in users:
        train_user_data = data_train[data_train['CustomerID'] == user]

        product_baskets = train_user_data[train_user_data['StockCode'] == product]['transaction_number'].unique()
        for basket in product_baskets:
            basket_items = train_user_data[train_user_data['transaction_number'] == basket]['StockCode']
            co_occurrence = list(set(basket_items) - {product})
            data_chunk[user - 1].setdefault(product, []).append(co_occurrence)


# Temporal dynamics
def temporal_decay_sum_history(data_chunk, item_size, within_decay_rate):
    sum_history = [{} for _ in range(user_size)]

    for user_index, user in enumerate(data_chunk):
        for key, co_items in user.items():
            num_baskets = len(co_items)
            decay_vec = np.arange(num_baskets, 0, -1, dtype=np.float16) ** within_decay_rate
            flat_items = np.concatenate(co_items)
            his_matrix = np.zeros(item_size, dtype=np.float16)

            start_idx = 0
            for weight, basket in zip(decay_vec, co_items):
                end_idx = start_idx + len(basket)
                his_matrix[flat_items[start_idx:end_idx] - 1] += weight
                start_idx = end_idx

            sum_history[user_index][key] = his_matrix

    return sum_history

# zero-shot
def max_k(search_set, item, item_for_user, num_nearest_neighbors,data_chuck,alpha):
    count = [len(list(chain(*data_chuck[user - 1][item]))) for user in item_for_user]
    count_cupy = cp.asarray(count)
    count_index = cp.argsort(count_cupy)[-num_nearest_neighbors:][::-1]
    user_popular = cp.array(item_for_user)[count_index]

    sets = cp.empty((len(user_popular), item_size), dtype=cp.float16)

    for i, user in enumerate(user_popular):
        sets[i] = cp.asarray(search_set[int(user - 1)][item])

    # get average
    merge_history = cp.asnumpy(((cp.sum(sets, axis=0)
                                 / len(count_index)) * (1 - alpha)).astype(np.float16))
    return merge_history

def cosine_similarity_vector_matrix(vector, matrix):
    vector_magnitude = cp.linalg.norm(vector)
    matrix_magnitudes = cp.linalg.norm(matrix, axis=1)
    dot_products = cp.dot(matrix, vector)

    cosine_similarities = dot_products / (vector_magnitude * matrix_magnitudes)

    return cosine_similarities


def KNN(target_set, vector,item, k, item_for_user, alpha):

    vector = cp.asarray(vector)
    selected_set = cp.empty((len(item_for_user), item_size), dtype=cp.float16)

    for i, user in enumerate(item_for_user):
        selected_set[i] = cp.asarray(target_set[(user - 1)][item])
    selected_set_transformed = (selected_set > 0).astype(int)
    vector_transformed = (vector > 0).astype(int)
    similarities = cosine_similarity_vector_matrix(vector_transformed, selected_set_transformed)
    indices = cp.argsort(similarities, kind='stable')[::-1][:k+1]

    history = cp.sum(selected_set[indices[1:]], axis=0)
    # for no neighbor
    count = len(indices[1:])+0.001

    merge_history = cp.asnumpy((vector * alpha + (history / count) * (1 - alpha)).astype(np.float16))

    return merge_history

# get score
temporal_decay_sum_history_training = temporal_decay_sum_history(data_chunk, item_size,group_decay_rate)
temporal_decay_new = copy.copy(temporal_decay_sum_history_training)

# renew score
for user, item in user_to_item_test_dict.items():
    for key in item:
        if key not in data_dict.keys():
            temporal_decay_new[user - 1][key] = np.zeros([item_size])

        elif key not in temporal_decay_sum_history_training[user - 1].keys():
            item_for_user = data_dict[key]  # all users who have the item
            temporal_decay_new[user - 1][key] = \
                max_k(temporal_decay_sum_history_training, key, item_for_user, num_nearest_neighbors,
                      data_chunk, alpha)
        elif key in temporal_decay_sum_history_training[user - 1].keys():
            vector = temporal_decay_sum_history_training[user - 1][key]
            item_for_user = data_dict[key]  # all users who have the item
            temporal_decay_new[user - 1][key] = \
                KNN(temporal_decay_sum_history_training, vector, key, num_nearest_neighbors, item_for_user, alpha)
print('finishing vector generation---------')

# -------------------start test------------------

hit_num, mrr_num = 0, 0

for user, items in user_to_item_test_dict.items():
    for target_item in items:
        given_item = np.setdiff1d(items, target_item)
        output_vectors = np.stack([temporal_decay_new[user - 1][kk] for kk in given_item])

        # Zero out the corresponding columns
        output_vectors[:, (given_item - 1)] = 0

        history_score = user_item_matrix[user - 1].copy()
        history_score[(given_item - 1)] = 0

        # Compute the combined score matrix
        output_add = ((1 - history_rate) * output_vectors + history_rate * history_score).astype(np.float16)

        # Get the top indices for recommendation
        flattened_matrix = output_add.flatten()
        sorted_indices = np.argsort(flattened_matrix)[::-1]
        seen_cols = set()
        top_indices = [col for idx in sorted_indices if
                       (col := np.unravel_index(idx, output_add.shape)[1]) not in seen_cols and not seen_cols.add(col)][
                      :topk]

        # Check if target item is in top 20
        if target_item - 1 in top_indices[:20]:
            hit_num += 1
            rank = top_indices[:20].index(target_item - 1) + 1
            mrr_num += 1 / rank

HR_20 = hit_num / test_number
MRR_20 = mrr_num / test_number

with open('results.txt', 'a', encoding='utf-8') as file:
    file.write('The number of neighbors is {}，α is {}，recall is： {}\n'.format(num_nearest_neighbors, alpha, HR_20))
    file.write('The number of neighbors is {}，α is {}，recall is： {}\n'.format(num_nearest_neighbors, alpha, MRR_20))
file.close()