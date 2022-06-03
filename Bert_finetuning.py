import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import label_ranking_average_precision_score
import random
import os 
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "true"
random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

## data

unsmile_data = pd.read_csv('data/unsmile_train_v1.0.tsv', delimiter='\t')
private_data = pd.read_csv('data/privatee.tsv', delimiter='\t',header = None)
private_data.columns = ['문장', 'name', 'number', 'address', 'bank', 'person']

private_data=private_data.sample(frac=1).reset_index(drop=True)
private_data = private_data[:int(len(private_data)*0.8)]
private_test_data = private_data[int(len(private_data)*0.8):]

concat_data = pd.concat([unsmile_data,private_data], axis=0)

concat_data = concat_data.fillna(0.0)

# print(concat_data)

# print(concat_data.info())


concat_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean", 'name', 'number', 'address', 'bank', 'person']

concat_num_labels = len(concat_labels)
# private_num_labels = len(private_labels)


concat_data_labels_df = concat_data[concat_labels]
concat_data_labels_df = concat_data_labels_df.astype(float)

# private_labels_df = private_data[private_labels]
# private_labels_df = private_labels_df.astype(float)

concat_label = []

for i in range(len(concat_data["여성/가족"])):
    concat_label.append([np.array(concat_data_labels_df.iloc[i], dtype=np.float64)])

print("===================================")
print(len(concat_label))

concat_label_data = pd.DataFrame(concat_label, columns = ['label'])

concat_data.rename(columns={'문장':'sentence'}, inplace=True)
concat_data.reset_index(inplace=True)


print(concat_label_data)
print(concat_data)



concat_dataset = pd.concat([concat_data['sentence'], concat_label_data['label']], axis=1)


# print(concat_dataset)


# private_label = []

# for i in range(len(private_data["name"])):

#     private_label.append([np.array(private_labels_df.iloc[i], dtype=np.float64)])

# pr_label_data = pd.DataFrame(private_label, columns = ['label'])

# private_data.rename(columns={'text':'sentence'}, inplace=True)

# private_dataset = pd.concat([private_data['sentence'], pr_label_data['label']], axis=1)


print(unsmile_data)
print(unsmile_data.info())
# print(private_data)
# print(private_data.info())



# model_name = 'beomi/kcbert-base'

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = BertForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=10,
#     problem_type="multi_label_classification"
# ).cuda()
# model.config.id2label = {i: label for i, label in zip(range(num_labels), unsmile_labels)}
# model.config.label2id = {label: i for i, label in zip(range(num_labels), unsmile_labels)}

# wandb.init(project="Hate_Speach", name=f"{model_name}-finetuning")


# train_data = unsmile_dataset[:int(len(unsmile_dataset)*0.8)]
# eval_data = unsmile_dataset[int(len(unsmile_dataset)*0.8):]

# train_text, train_labels = (
#     train_data["sentence"].values,
#     train_data["label"].values,
# )

# # print(train_labels)

# dataset = [
#     {"data": t, "label": l}
#     for t, l in zip(train_text, train_labels)
# ]

# train_loader = DataLoader(
#     dataset,
#     batch_size=64,
#     num_workers=8,
#     drop_last=True,
#     pin_memory=True,
# )


# # eval_data = pd.read_csv("data/train_data.csv", delimiter="\t")
# # eval_data = eval_data[100000:]
# eval_text, eval_labels = (
#     eval_data["sentence"].values,
#     eval_data["label"].values
# )

# dataset = [
#     {"data": t, "label": l}
#     for t, l in zip(eval_text, eval_labels)
# ]
# eval_loader = DataLoader(
#     dataset,
#     batch_size=64,
#     num_workers=8,
#     drop_last=True,
#     pin_memory=True,
# )

# optimizer = AdamW(params=model.parameters(),
#     lr=3e-5, weight_decay=3e-7
# )

# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     for train in tqdm(train_loader):
#         optimizer.zero_grad()
#         text, label = train["data"], train["label"].cuda()
#         tokens = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#         )

#         input_ids = tokens.input_ids.cuda()
#         attention_mask = tokens.attention_mask.cuda()
#         output = model.forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=label
#         )

#         loss = output.loss
#         loss.backward()        
#         optimizer.step()
#         classification_results = output.logits
        
#         # lrap_score = 0
#         # lrap_score += label_ranking_average_precision_score(label.detach().cpu().numpy(), classification_results.detach().cpu().numpy())
#         # print(label_ranking_average_precision_score(label.detach().cpu().numpy(), classification_results.detach().cpu().numpy()))
#         # print(len(classification_results))
#     print({'train_lrap': label_ranking_average_precision_score(label.detach().cpu().numpy(), classification_results.detach().cpu().numpy())})
#     print({"train_loss": loss.item()})
#     wandb.log({"train_lrap": label_ranking_average_precision_score(label.detach().cpu().numpy(), classification_results.detach().cpu().numpy())})
#     wandb.log({"train_loss": loss.item()})
#     # print({"acc": acc / len(classification_results)})   ## 탭하나 안에 넣으면 step단위로 볼수있음. 

#     with torch.no_grad():
#         model.eval() 
#         for eval in tqdm(eval_loader):
#             eval_text, eval_label = eval["data"], eval["label"].cuda()
#             eval_tokens = tokenizer(
#                 eval_text,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding=True,
#             )

#             input_ids = eval_tokens.input_ids.cuda()
#             attention_mask = eval_tokens.attention_mask.cuda()

#             eval_out = model.forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=eval_label
#             )

#             eval_loss = eval_out.loss
#             eval_classification_results = eval_out.logits

#         print({'eval_lrap': label_ranking_average_precision_score(eval_label.detach().cpu().numpy(), eval_classification_results.detach().cpu().numpy())})
#         print({"eval_loss": eval_loss.item()})
#         print({"epoch": epoch+1})

#         wandb.log({"eval_lrap": label_ranking_average_precision_score(eval_label.detach().cpu().numpy(), eval_classification_results.detach().cpu().numpy())})
#         wandb.log({"eval_loss": eval_loss.item()})
#         wandb.log({"epoch": epoch+1})

#         torch.save(model.state_dict(), f"model_save/Hate_Speach-{model_name.replace('/', '-')}-{epoch+1}.pt")
