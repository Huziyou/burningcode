import pandas as pd
import numpy as np

train_in_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/train_in.csv"
train_out_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/train_out.csv"
test_in_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/test_in.csv"
test_out_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/test_out.csv"

image_data = pd.read_csv(train_in_file_path, header=None)
image_labels = pd.read_csv(train_out_file_path, header=None)
image_test = pd.read_csv(test_in_file_path,header = None)
valification = pd.read_csv(test_out_file_path,header=None)

combined_data = pd.concat([image_data, image_labels], axis=1, ignore_index=True)
combined_data.to_csv('combine_data.csv', index=False, header=False)

data_labels_dict = {}
column_means_dict = {}
vector_means_dict = {}

for label in range(10):
    data_labels_dict[label] = combined_data[combined_data[combined_data.columns[-1]] == label]
    column_means_dict[label] = data_labels_dict[label].mean()

    vector_means_dict[label] = column_means_dict[label].iloc[:-1].tolist()
    
    df = pd.DataFrame([vector_means_dict[label]])
    df.to_csv(f'means_{label}.csv', index=False, header=False)
    
valification_list = valification.iloc[:,0].tolist()
"""
predict_class = []
test_in_rows = image_test.shape[0]
for i in range(test_in_rows):
    test_image_vector = image_test.iloc[i].tolist()
    set_distances = {}
        

    for label, mean_vector in vector_means_dict.items():
        distance = np.linalg.norm(np.array(test_image_vector) - np.array(mean_vector))
        set_distances[label] = distance
    predict_class.append(min(set_distances, key= set_distances.get))


accuracy_number = 0
error_number =0

for i in range(len(valification_list)):
    if valification_list[i] == predict_class[i]:
        accuracy_number += 1
    else:
        error_number += 1
accuracy = accuracy_number/(accuracy_number+error_number)

print(accuracy)
"""
# ... [您之前的代码]

# 初始化正确预测和总预测的计数字典
correct_predictions_per_class = {label: 0 for label in range(10)}
total_predictions_per_class = {label: 0 for label in range(10)}

predict_class = []
test_in_rows = image_test.shape[0]

for i in range(test_in_rows):
    test_image_vector = image_test.iloc[i].tolist()
    set_distances = {}

    for label, mean_vector in vector_means_dict.items():
        distance = np.linalg.norm(np.array(test_image_vector) - np.array(mean_vector))
        set_distances[label] = distance

    predicted_label = min(set_distances, key=set_distances.get)
    predict_class.append(predicted_label)

    true_label = valification_list[i]
    total_predictions_per_class[true_label] += 1
    
    if true_label == predicted_label:
        correct_predictions_per_class[true_label] += 1

# 计算总体准确率
accuracy = sum(correct_predictions_per_class.values()) / sum(total_predictions_per_class.values())
print(f"Overall Accuracy: {accuracy}")

# 计算并打印每个类别的准确率
for label in range(10):
    if total_predictions_per_class[label] != 0:  # 避免除以零的错误
        accuracy_per_class = correct_predictions_per_class[label] / total_predictions_per_class[label]
    else:
        accuracy_per_class = 0
    print(f"Accuracy for class {label}: {accuracy_per_class}")
