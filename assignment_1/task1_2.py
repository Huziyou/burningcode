import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_in_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/train_in.csv"
train_out_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/train_out.csv"
test_in_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/test_in.csv"
test_out_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/test_out.csv"


image_data = pd.read_csv(train_in_file_path, header=None)
image_labels = pd.read_csv(train_out_file_path, header=None)
image_test = pd.read_csv(test_in_file_path,header = None)
valification = pd.read_csv(test_out_file_path,header=None)


means_train_in = np.mean(image_data,axis =0)
centered_data_means = image_data - means_train_in
cov_matrix = np.dot(centered_data_means.T, centered_data_means) / (centered_data_means.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
k = np.where(cumulative_variance_ratio >= 0.9)[0][0] + 1
top_k_eigenvectors = eigenvectors[:, :k]
reduced_data = np.dot(centered_data_means, top_k_eigenvectors)

pd.DataFrame(reduced_data).to_csv('reduced_data_PCA.csv', index=False, header=False)


reduced_data_df = pd.DataFrame(reduced_data)
combined_data = pd.concat([reduced_data_df, image_labels], axis=1, ignore_index=True)
combined_data.to_csv('reduced_data_combined.csv', index=False, header=False)
print_combined_data = pd.concat([reduced_data_df, image_labels], axis=1, ignore_index=True)

#为降二维的数据画图
print_k = 2
print_top_k_eigenvectors = eigenvectors[:, :print_k]
print_data = np.dot(centered_data_means, print_top_k_eigenvectors)
print_data_labels_dict = {}
print_column_means_dict = {}
print_vector_means_dict = {}
print_data_df = pd.DataFrame(print_data)
for label in range(10):
    print_data_labels_dict[label] = print_combined_data[print_combined_data[print_combined_data.columns[-1]] == label]
    print_column_means_dict[label] = print_data_labels_dict[label].mean()
    print_vector_means_dict[label] = print_column_means_dict[label].iloc[:-1].tolist()

plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for label, color in zip(print_data_labels_dict.keys(), colors):
    x_vals = print_data_labels_dict[label].iloc[:, 0].values
    y_vals = print_data_labels_dict[label].iloc[:, 1].values
    plt.scatter(x_vals, y_vals, c=color, label=f"Class {label}", alpha=0.5)
    
plt.legend()
plt.title('PCA Reduced Data Scatter Plot by Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')



data_labels_dict = {}
column_means_dict = {}
vector_means_dict = {}

for label in range(10):
    data_labels_dict[label] = combined_data[combined_data[combined_data.columns[-1]] == label]
    column_means_dict[label] = data_labels_dict[label].mean()
    vector_means_dict[label] = column_means_dict[label].iloc[:-1].tolist()
    df = pd.DataFrame([vector_means_dict[label]])
    df.to_csv(f'means_{label}.csv', index=False, header=False)




distance_cloud = {}
for label1 in range(10):
    for label2 in range(label1+1, 10): 
        distance = np.linalg.norm(np.array(vector_means_dict[label1]) - np.array(vector_means_dict[label2]))
        key = (label1, label2)
        distance_cloud[key] = distance
for key, value in distance_cloud.items():
    print(f"Distance between class {key[0]} and class {key[1]}: {value}")


valification_list = valification.iloc[:,0].tolist()

correct_predictions_per_class = {label: 0 for label in range(10)}
total_predictions_per_class = {label: 0 for label in range(10)}

for i in range(image_test.shape[0]):
    original_test_image_vector = image_test.iloc[i].tolist()
    centered_test_image_vector = np.array(original_test_image_vector) - means_train_in
    test_image_vector = np.dot(centered_test_image_vector, top_k_eigenvectors)
    
    set_distances = {}
    for label, mean_vector in vector_means_dict.items():
        distance = np.linalg.norm(test_image_vector - np.array(mean_vector))
        set_distances[label] = distance

    predicted_label = min(set_distances, key=set_distances.get)
    true_label = valification_list[i]
    total_predictions_per_class[true_label] += 1
    
    if true_label == predicted_label:
        correct_predictions_per_class[true_label] += 1

accuracy = sum(correct_predictions_per_class.values()) / sum(total_predictions_per_class.values())
print(f"Overall Accuracy: {accuracy}")

for label in range(10):
    if total_predictions_per_class[label] != 0: 
        accuracy_per_class = correct_predictions_per_class[label] / total_predictions_per_class[label]
    else:
        accuracy_per_class = 0
    print(f"Accuracy for class {label}: {accuracy_per_class}")



