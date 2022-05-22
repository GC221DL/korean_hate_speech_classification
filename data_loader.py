import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

train_data = pd.read_csv('data/unsmile_train_v1.0.tsv', delimiter='\t')

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]
num_labels = len(unsmile_labels)
# train_data['label'] = 0

labels_df = train_data[["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]]
labels_df = labels_df.astype(float)


# print(labels_df)

# print(train_data.tail())


# train_data['label'] = train_data[unsmile_labels].apply(lambda row: ','.join(row.values.astype(str)), axis=1)

# train_data['label'] = torch.tensor(train_data["label"], dtype=torch.float)

label = []

for i in range(len(train_data["여성/가족"])):

    label.append([list(labels_df.iloc[i])])

# print(label)

label_data = pd.DataFrame(label, columns = ['label'])

train_data.rename(columns={'문장':'sentence'}, inplace=True)

# label_data['label']= label_data["label"].replace("[", "").replace("[", "")

dataset = pd.concat([train_data['sentence'], label_data['label']], axis=1)


# dataset.to_csv('data/train_data.csv', sep='\t', index = False)

train_labels = dataset["label"].replace("[", "").replace("[", "")

train_text, train_labels = (
    dataset["sentence"].values,
    dataset["label"].values,
)

print(train_labels)
# list2 = sum(train_labels, [])
# print(list2)



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

# for i in train_loader:
#     print(i)



# print(label_data, column = ['label'])

# print(train_data[0])
# label_dict = {
#             "여성/가족" : 0, 
#             "남성" : 1, 
#             "성소수자" : 2,
#             "인종/국적" : 3,
#             "연령" : 4, 
#             "지역" : 5,
#             "종교" : 6,
#             "기타 혐오" : 7,
#             "악플/욕설" : 8,
#             "clean" : 9
#             }

# train_data["label"] = train_data["label"].replace(label_dict)

# train_data['여성/가족'] = train_data['여성/가족'] == '1'


# df_label = pd.concat(["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"], axis = 0)
# print(train_data.info())
# print(train_data.head())

# id2label = {i: label for i, label in zip(range(num_labels), unsmile_labels)}
# label2id = {label: i for i, label in zip(range(num_labels), unsmile_labels)}

