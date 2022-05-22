import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

train_data = pd.read_csv('data/unsmile_valid_v1.0.tsv', delimiter='\t')

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]
num_labels = len(unsmile_labels)

labels_df = train_data[["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]]
labels_df = labels_df.astype(float)

label = []

for i in range(len(train_data["여성/가족"])):

    label.append([np.array(labels_df.iloc[i], dtype=np.float64)])

# print(label)

label_data = pd.DataFrame(label, columns = ['label'])

train_data.rename(columns={'문장':'sentence'}, inplace=True)

dataset = pd.concat([train_data['sentence'], label_data['label']], axis=1)

dataset.to_csv('data/test_data.csv', sep='\t', index = False)

# train_data = pd.read_csv("data/train_data_test.csv", delimiter="\t")

# train_data = dataset[:int(len(dataset)*0.8)]
# eval_data = dataset[int(len(dataset)*0.8):]

# train_text, train_labels = (
#     train_data["sentence"].values,
#     train_data["label"].values,
# )

# # print(train_labels)

# # train_labels = np.array(train_labels, dtype=np.float64)

# dataset = [
#     {"data": t, "label": l}
#     for t, l in zip(train_text, train_labels)
# ]

# train_loader = DataLoader(
#     dataset,
#     batch_size=1,
#     num_workers=8,
#     drop_last=True,
#     pin_memory=True,
# )

# for train in train_loader:
#     text, label = train["data"], train["label"]
#     print(label)
