import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import time
from itertools import combinations
import collections
import math
import random

def load_data_dcs(file_path):
    data = np.load(file_path)
    data[:, 2] = 1
    user_ids = data[:, 0]
    movie_ids = data[:, 1]
    ratings = data[:, 2]
    user_indices = np.array(user_ids) - 1
    movie_indices = np.array(movie_ids) - 1
    # Create sparse matrix using CSR format
    user_movie_matrix = csr_matrix((np.ones_like(user_ids), (user_indices, movie_indices)))
    num_movies = np.max(data[:, 1])
    return user_movie_matrix, num_movies

def generate_signatures(user_movie_matrix):
    # random projection
    num_hyperplanes = 100

    num_users = user_movie_matrix.shape[1]
    projection_matrix = np.random.randn(num_users, num_hyperplanes)
    projected_matrix = user_movie_matrix.dot(projection_matrix)
    user_hash_signatures = np.where(projected_matrix > 0, 1, 0)
    return user_hash_signatures


def allocate_bucket_dcs(user_hash_signatures):
    b = 6
    r = 15
    num_buckets = 40000
    # Initialize hash bucket
    buckets = {i: [] for i in range(num_buckets)}
    for user_id, signature in enumerate(user_hash_signatures):
        for band in range(b):
            start_row = band * r
            end_row = start_row + r
            band_signature = tuple(signature[start_row:end_row])
            hash_value = hash(band_signature) % num_buckets
            buckets[hash_value].append(user_id)
    return buckets


def calculate_adjusted_cosine_similarity_dcs(p1, p2):
    dot_product = np.dot(p1, p2)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)
    similarity = dot_product / (norm_p1 * norm_p2)
    similarity = np.clip(similarity, -1, 1)
    angle_in_degrees = np.arccos(similarity) * (180 / np.pi)
    adjusted_similarity = 1 - angle_in_degrees / 180
    return adjusted_similarity

# num_movies = np.max(data[:, 1])

# def original_rating_vector(user_id, data, num_movies):
#     user_ratings = data[data[:, 0] == user_id]
#     movie_indices = user_ratings[:, 1].astype(int) - 1
#     rating_vector = np.bincount(movie_indices, weights=user_ratings[:, 2], minlength=num_movies)
#     return rating_vector
       
# print(f"total: {len(buckets)} buckets")

def calculate_dcs(buckets, user_hash_signatures, user_movie_matrix):
    already_compared = set()
    similar_users = []
    time_limit = 28 * 60 
    start_time = time.time()

    for bucket in buckets.values():
        for user1, user2 in combinations(bucket, 2):
            if user1 == user2:
                continue
            if time.time() - start_time > time_limit:
                print("out of time")
                break
            if (user1, user2) not in already_compared and (user2, user1) not in already_compared:
                similarity = calculate_adjusted_cosine_similarity_dcs(user_hash_signatures[user1, :], user_hash_signatures[user2, :])
                already_compared.add((user1, user2))
                if similarity > 0.73:
                    # ori_user1 = original_rating_vector(user1+1, data, num_movies)
                    # ori_user2 = original_rating_vector(user2+1, data, num_movies)
                    ori_user1 = user_movie_matrix.getrow(user1).toarray().ravel()
                    ori_user2 = user_movie_matrix.getrow(user2).toarray().ravel()
                    real_similarity = calculate_adjusted_cosine_similarity_dcs(ori_user1, ori_user2)
                    if real_similarity > 0.73:
                        similar_users.append(tuple(sorted([user1, user2])))
        if time.time() - start_time > time_limit:
            break
        
    unique_similar_users = list(set(similar_users))
    return unique_similar_users


def set_random_seed(seed):
    np.random.seed(seed)

def write_file_dcs(unique_similar_users):
    with open('similar_users_dcs.txt', 'w') as file:
        for pair in unique_similar_users:
            file.write(f"{pair[0]}, {pair[1]}\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data file path')
    parser.add_argument('-s', type=int, default=17, help='Random seed')
    parser.add_argument("-m", "--measure", type=str, choices=["js", "cs", "dcs"], help="Similarity measure", required=True)

    args = parser.parse_args()
    if args.measure == 'dcs':
        user_movie_matrix, num_movies = load_data_dcs(args.d)
        set_random_seed(args.s)
        user_hash_signatures = generate_signatures(user_movie_matrix)
        buckets = allocate_bucket_dcs(user_hash_signatures)
        unique_similar_users = calculate_dcs(buckets, user_hash_signatures, user_movie_matrix)
        write_file_dcs(unique_similar_users)

if __name__ == "__main__":
    main()