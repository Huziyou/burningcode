import numpy as np
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
# from sklearn.random_projection import SparseRandomProjection
# from sklearn.metrics.pairwise import cosine_similarity
import time
import argparse
import numpy as np
from scipy import sparse
from itertools import combinations
import time
import argparse 

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

def generate_signatures_dcs(user_movie_matrix):
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


def set_random_seed_dcs(seed):
    np.random.seed(seed)

def write_file_dcs(unique_similar_users):
    with open('dcs.txt', 'w') as file:
        for pair in unique_similar_users:
            file.write(f"{pair[0]}, {pair[1]}\n")

# Load data
def load_data_js(file_path):
    data = np.load(file_path)
    user_id = data[:, 0]
    movie_id = data[:, 1]
    # Make sure the index begins with 0
    user_indices = np.array(user_id) - 1
    movie_indices = np.array(movie_id) - 1
    # Create sparse matrix using CSR format
    matrix = sparse.csr_matrix((np.ones_like(user_id), (user_indices, movie_indices)))
    return matrix


def generate_signature_js(matrix):
    # Initialize signature matrix
    num_users, num_movies = matrix.shape
    num_permutations = 100
    signatures = np.full((num_users, num_permutations), np.inf)
    # Pre-compute the indices of rated movies for each user
    user_rated_movies = [matrix.getrow(i).nonzero()[1] for i in range(num_users)]

    for p in range(num_permutations):
        # Generate a random permutation of movie indices
        permuted_indices = np.random.permutation(num_movies)
        # Invert the permutation to get the new positions of each movie index
        inverted_permutation = np.argsort(permuted_indices)
    
        for user_id, movies in enumerate(user_rated_movies):
            # Apply the inverted permutation to the user's rated movie indices
            new_positions = inverted_permutation[movies]
            # The smallest new position corresponds to the first movie in the permuted order
            min_new_position = np.min(new_positions)
            # Update the signature matrix
            signatures[user_id, p] = min_new_position
    return signatures, user_rated_movies


def allocate_bucket_js(signatures):
    b = 10
    r = 4
    num_buckets = 31500
    # Initialize hash bucket
    buckets = {i: [] for i in range(num_buckets)}
    for user_id, signature in enumerate(signatures):
        for band in range(b):
            start_row = band * r
            end_row = start_row + r
            band_signature = tuple(signature[start_row:end_row])
            # Same signature has same hash value 
            hash_value = hash(band_signature) % num_buckets
            # Add user id into bucket
            buckets[hash_value].append(user_id)
    return buckets

def set_random_seed_js(seed):
    np.random.seed(seed)

def estimate_jaccard_similarity(signature1, signature2):
    matches = np.sum(signature1 == signature2)
    return matches / len(signature1)


def calculate_jaccard_similarity(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0



def calculation_js(buckets, signatures, user_rated_movies):
    time_limit = 30 * 60 
    start_time = time.time() 
    threshold = 0.5
    max_bucket_size = 5000  
    total_candidate_pairs = 0
    successful_pairs = 0
    time_limit_reached = False
    
    output_file_path = 'js.txt'
    # Open file 
    with open(output_file_path, 'w') as file:
        # Traverse the list of all user_id in the bucket
        for bucket in buckets.values():
            if len(bucket) > max_bucket_size:
                continue
            for user1, user2 in combinations(bucket, 2):
                if user1 == user2:
                    continue
                if time.time() - start_time > time_limit:
                    print("Hitting the time limit, further calculations are halted.")
                    time_limit_reached = True 
                    # Jump out of an inner loop
                    break
                total_candidate_pairs += 1
                jaccard_similarity = estimate_jaccard_similarity(signatures[user1], signatures[user2])
                if jaccard_similarity > threshold:
                    similarity = calculate_jaccard_similarity(user_rated_movies[user1], user_rated_movies[user2])
                    if similarity > threshold:
                        successful_pairs += 1
                        file.write(f'{user1}, {user2}\n')
            # Jump out of an outer loop
            if time_limit_reached:
                break  

def signature_rating_vector(user_id1, user_id2, signatures):
    vector1 = signatures[user_id1].flatten()
    vector2 = signatures[user_id2].flatten()
    return vector1,vector2

def original_rating_vector(user_id, data, num_movies):
    rating_vector = np.zeros(num_movies)
    user_ratings = data[data[:, 0] == user_id]  
    for _, movie_id, rating in user_ratings:
        rating_vector[movie_id - 1] = rating 
    return rating_vector

def calculate_adjusted_cosine_similarity(vector1, vector2):
    cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    theta = np.arccos(np.clip(cos_sim, -1, 1))  
    adjusted_cos_sim = 1 - (theta / np.pi) 
    return adjusted_cos_sim


def load_data_cs(file_path):
    data = np.load(file_path)
    user_ids = data[:, 0].astype(int)
    movie_ids = data[:, 1].astype(int)
    ratings = data[:, 2]
    num_users = user_ids.max() + 1
    num_movies = movie_ids.max() + 1
    return csr_matrix((ratings, (user_ids, movie_ids)), shape=(num_users, num_movies)), data, num_movies

def set_random_seed_cs(seed):
    np.random.seed(seed)


def allocate_bucket_cs(signatures):
    r = 15
    b = 6
    num_buckets = 31700
    buckets = {i: [] for i in range(num_buckets)}
    for user_id, signature in enumerate(signatures):
        for band in range(b):
            start_row = band * r
            end_row = start_row + r
            band_signature = tuple(signature[start_row:end_row])
            hash_value = hash(band_signature) % num_buckets
            buckets[hash_value].append(user_id)
    return buckets


def calculation_cos(buckets, signatures, data, num_movies):
    similar_users_set = set()
    calculated_pairs = set()
    start_time = time.time()
    max_duration = 29 * 60 
    timeout_occurred = False
    large_bucket_threshold = 57630 
    large_buckets = []  
    calculated_pairs = set()  
    similar_users_set = set() 

    for bucket_users in buckets.values():
        if len(bucket_users) * (len(bucket_users) - 1) / 2 > large_bucket_threshold:
            large_buckets.append(bucket_users)
            continue
        for i in range(len(bucket_users)):
            if time.time() - start_time > max_duration:
                print("time over.")
                timeout_occurred = True
                break
            for j in range(i + 1, len(bucket_users)):
                user_id1 = bucket_users[i]
                user_id2 = bucket_users[j]
                user_pair = (min(user_id1, user_id2), max(user_id1, user_id2))
                if user_pair in calculated_pairs:
                    continue
                if user_id1 != user_id2:
                    sig_user1, sig_user2 = signature_rating_vector(user_id1, user_id2, signatures)
                    similarity_sig = calculate_adjusted_cosine_similarity(sig_user1, sig_user2)
                    if similarity_sig > 0.73:
                        user_id1_original = original_rating_vector(user_id1, data, num_movies)
                        user_id2_original = original_rating_vector(user_id2, data, num_movies)
                        similarity_original = calculate_adjusted_cosine_similarity(user_id1_original, user_id2_original)
                        calculated_pairs.add(user_pair)
                        if similarity_original > 0.73:
                            similar_users_set.add(user_pair)
                            
                            print(similarity_original)
                            
        if timeout_occurred:
            break
    

    if not timeout_occurred:
        for bucket_users in large_buckets:
            for i in range(len(bucket_users)):
                if time.time() - start_time > max_duration:
                    print("time over.")
                    break
                for j in range(i + 1, len(bucket_users)):
                    user_id1 = bucket_users[i]
                    user_id2 = bucket_users[j]
                    user_pair = (min(user_id1, user_id2), max(user_id1, user_id2))
                    if user_pair in calculated_pairs:
                        continue
                    if user_id1 != user_id2:
                        sig_user1, sig_user2 = signature_rating_vector(user_id1, user_id2, signatures)
                        similarity_sig = calculate_adjusted_cosine_similarity(sig_user1, sig_user2)
                        if similarity_sig > 0.73:
                            user_id1_original = original_rating_vector(user_id1, data, num_movies)
                            user_id2_original = original_rating_vector(user_id2, data, num_movies)
                            similarity_original = calculate_adjusted_cosine_similarity(user_id1_original, user_id2_original)
                            calculated_pairs.add(user_pair)
                            if similarity_original > 0.73:
                                similar_users_set.add(user_pair)
                                
                                print(similarity_original)
                                

    print("total find:", len(similar_users_set))
    return similar_users_set

def output_write_cs(similar_users_set):
    output_file_path = 'cs.txt'
    with open(output_file_path, 'w') as file:
        for u1, u2 in similar_users_set:
            file.write(f"{u1},{u2}\n")

    print(f"finish writing {output_file_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data file path')
    parser.add_argument('-s', type=int, default=17, help='Random seed')
    parser.add_argument("-m", "--measure", type=str, choices=["js", "cs", "dcs"], help="Similarity measure", required=True)

    args = parser.parse_args()
    if args.measure == 'cs':
        ratings_matrix, data, num_movies = load_data_cs(args.d)
        set_random_seed_cs(args.s)
        num_hyperplanes = 100
        random_hyperplanes = np.random.randn(num_movies, num_hyperplanes)
        signatures = (ratings_matrix.dot(random_hyperplanes) >= 0).astype(int) * 2 - 1
        buckets = allocate_bucket_cs(signatures)
        similar_users_set = calculation_cos(buckets, signatures, data, num_movies)
        output_write_cs(similar_users_set)

    if args.measure == 'js':
        matrix = load_data_js(args.d)
        set_random_seed_js(args.s)
        signatures, user_rated_movies = generate_signature_js(matrix)
        buckets = allocate_bucket_js(signatures)
        calculation_js(buckets, signatures, user_rated_movies)
    
    if args.measure == 'dcs':
        user_movie_matrix, num_movies = load_data_dcs(args.d)
        set_random_seed_dcs(args.s)
        user_hash_signatures = generate_signatures_dcs(user_movie_matrix)
        buckets = allocate_bucket_dcs(user_hash_signatures)
        unique_similar_users = calculate_dcs(buckets, user_hash_signatures, user_movie_matrix)
        write_file_dcs(unique_similar_users)


if __name__ == "__main__":
    main()
