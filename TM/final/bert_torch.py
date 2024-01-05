import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, recall_score

# Early Stopping Helper
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Custom BERT Model for Sequence Classification
class BertForSequenceClassificationCustom(nn.Module):
    def __init__(self, num_labels=3, dropout_rate=0.6):
        super(BertForSequenceClassificationCustom, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data Preprocessing Functions
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text

def tokenize_text(text, tokenizer, max_length=128):
    return tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )

# Read and Prepare Data
def read_and_prepare_data(file_path, tokenizer):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'sentiment', 'text'])
    df['text'] = df['text'].apply(clean_text)
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)
    df['tokenized'] = df['text'].apply(lambda x: tokenize_text(x, tokenizer))

    labels = torch.tensor(df['sentiment'].values)
    input_ids = torch.cat([x['input_ids'] for x in df['tokenized'].values], dim=0)
    attention_masks = torch.cat([x['attention_mask'] for x in df['tokenized'].values], dim=0)

    return TensorDataset(input_ids, attention_masks, labels)

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Read and Process Data
train_dataset = read_and_prepare_data('/home/s3963616/TM/final/data/twitter-2013train-A.tsv', tokenizer)
val_dataset = read_and_prepare_data('/home/s3963616/TM/final/data/twitter-2013dev-A.tsv', tokenizer)
test_dataset = read_and_prepare_data('/home/s3963616/TM/final/data/twitter-2013test-A.tsv', tokenizer)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model Initialization
model = BertForSequenceClassificationCustom(num_labels=3, dropout_rate=0.3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
print(model)

# Training Loop with Early Stopping
epochs = 10
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = tuple(b.to(device) for b in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        logits = model(b_input_ids, attention_mask=b_input_mask)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 3), b_labels.view(-1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch}, Average Training Loss: {avg_train_loss}')

    model.eval()
    total_eval_loss = 0
    for batch in validation_dataloader:
        batch = tuple(b.to(device) for b in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask)
        loss = loss_fct(logits.view(-1, 3), b_labels.view(-1))
        total_eval_loss += loss.item()
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print(f'Epoch {epoch}, Validation Loss: {avg_val_loss}')

    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Test Evaluation
model.eval()
total_test_accuracy = 0
total_test_loss = 0
all_preds = []
all_label_ids = []

for batch in test_dataloader:
    batch = tuple(b.to(device) for b in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        logits = model(b_input_ids, attention_mask=b_input_mask)
    loss = loss_fct(logits.view(-1, 3), b_labels.view(-1))
    total_test_loss += loss.item()
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    preds = np.argmax(logits, axis=1)
    all_preds.extend(preds)
    all_label_ids.extend(label_ids)

avg_test_loss = total_test_loss / len(test_dataloader)
avg_test_accuracy = np.mean(np.array(all_preds) == np.array(all_label_ids))


print(f'Test Loss: {avg_test_loss}')
print(f'Test Accuracy: {avg_test_accuracy}')
print(classification_report(all_label_ids, all_preds, target_names=['negative', 'neutral', 'positive']))


# 计算各类别召回率
recall_positive = recall_score(all_label_ids, all_preds, labels=[2], average='macro')
recall_negative = recall_score(all_label_ids, all_preds, labels=[0], average='macro')
recall_neutral = recall_score(all_label_ids, all_preds, labels=[1], average='macro')

print(f"Recall for Positive (RP): {recall_positive}")
print(f"Recall for Negative (RN): {recall_negative}")
print(f"Recall for Neutral (RU): {recall_neutral}")


average_recall = (recall_positive + recall_negative + recall_neutral) / 3
print(f"Average Recall: {average_recall}")



# print(df[['sentiment', 'text']].head())


# sample_texts = df['text'].sample(5).tolist()
# for text in sample_texts:
#     print("原始文本:", text)
#     print("分词结果:", tokenizer.tokenize(text))
#     print("编码结果:", tokenizer.encode(text, add_special_tokens=True))
#     print()


# lengths = [len(input) for input in input_ids]
# print("序列长度:", set(lengths))

# # 检查注意力掩码中是否只有0和1
# unique_values_in_masks = set(attention_masks.numpy().flatten())
# assert unique_values_in_masks.issubset({0, 1}), "注意力掩码中只能包含0和1"

# print("input_ids 形状:", input_ids.shape)
# print("attention_masks 形状:", attention_masks.shape)
# print("labels 形状:", labels.shape)