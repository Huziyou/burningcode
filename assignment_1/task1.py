import pandas as pd

train_in_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/train_in.csv"
train_out_file_path = "/Users/wushiran/Desktop/assignment/IDL/assignment_1/train_out.csv"

image_data = pd.read_csv(train_in_file_path, header=None)
image_labels = pd.read_csv(train_out_file_path, header=None)

combined_data = pd.concat([image_data, image_labels], axis=1, ignore_index=True)
combined_data.to_excel('combine_data.xlsx', index=False, header=False)

data_labels_dict = {}
column_means_dict = {}
vector_means_dict = {}

for label in range(10):
    data_labels_dict[label] = combined_data[combined_data[combined_data.columns[-1]] == label]
    column_means_dict[label] = data_labels_dict[label].mean()

    vector_means_dict[label] = column_means_dict[label].iloc[:-1].tolist()
    
    df = pd.DataFrame([vector_means_dict[label]])
    df.to_excel(f'means_{label}.xlsx', index=False, header=False)
