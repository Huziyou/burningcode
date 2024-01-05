import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import numpy as np
from tensorflow import keras

# 步骤 1: 读取数据
train_data = pd.read_csv('/home/s3963616/TM/final/data/twitter-2016train-A.tsv', sep='\t', header=None)
test_data = pd.read_csv('/home/s3963616/TM/final/data/twitter-2016devtest-A.tsv', sep='\t', header=None)
dev_data = pd.read_csv('/home/s3963616/TM/final/data/twitter-2016dev-A.tsv', sep='\t', header=None)

# 提取文本和标签
train_tweets = train_data[2].astype(str).str.lower()
train_labels = train_data[1]
test_tweets = test_data[2].astype(str).str.lower()
test_labels = test_data[1]
dev_tweets = dev_data[2].astype(str).str.lower()
dev_labels = dev_data[1]

# 步骤 2: 标签编码
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)
train_labels_enc = label_encoder.transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)
dev_labels_enc = label_encoder.transform(dev_labels)
train_labels_enc = keras.utils.to_categorical(train_labels_enc)
test_labels_enc = keras.utils.to_categorical(test_labels_enc)
dev_labels_enc = keras.utils.to_categorical(dev_labels_enc)

# 步骤 3: 初始化BERT的Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 步骤 4: 文本转换为BERT所需格式
train_encodings = tokenizer(train_tweets.tolist(), truncation=True, padding='max_length', max_length=100, return_tensors="tf")
test_encodings = tokenizer(test_tweets.tolist(), truncation=True, padding='max_length', max_length=100, return_tensors="tf")
dev_encodings = tokenizer(dev_tweets.tolist(), truncation=True, padding='max_length', max_length=100, return_tensors="tf")

# 步骤 5: 创建TensorFlow数据集
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels_enc))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels_enc))
dev_dataset = tf.data.Dataset.from_tensor_slices((dict(dev_encodings), dev_labels_enc))

# 步骤 6: 定义和编译模型
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 步骤 7: 训练模型
model.fit(train_dataset.shuffle(100).batch(32), epochs=3, validation_data=dev_dataset.batch(32))

# 步骤 8: 模型评估
model.evaluate(test_dataset.batch(32))

# 假设 y_true 包含真实标签，y_pred 包含模型预测的标签
# 首先，我们需要将独热编码的标签转换回原始的类别编码
y_true = np.argmax(test_labels_enc, axis=1)
y_pred = model.predict(test_encodings)
y_pred = np.argmax(y_pred, axis=1)

# 计算每个类别的F1分数
f1_positive = f1_score(y_true, y_pred, labels=[label_encoder.transform(['positive'])[0]], average='macro')
f1_negative = f1_score(y_true, y_pred, labels=[label_encoder.transform(['negative'])[0]], average='macro')
f1_neutral = f1_score(y_true, y_pred, labels=[label_encoder.transform(['neutral'])[0]], average='macro')


from sklearn.metrics import classification_report

# Assume y_true and y_pred are already defined
# y_true are the true class labels
# y_pred are the predicted class labels by the model

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

print(report)

# 计算平均F1分数
f1_pn = (f1_positive + f1_negative + f1_neutral) / 3

print('F1 Score for Neutral class:', f1_neutral)
print('F1 Score for Positive class:', f1_positive)
print('F1 Score for Negative class:', f1_negative)