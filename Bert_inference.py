"""
python inference.py
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, ElectraForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import label_ranking_average_precision_score
import os
import random

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
unsmile_data = pd.read_csv('data/unsmile_valid_v1.0.tsv', delimiter='\t')  
private_data = pd.read_csv('data/private_test.tsv', delimiter='\t',header = None)
private_data.columns = ['문장', 'name', 'number', 'address', 'bank', 'person']

private_data=private_data.sample(frac=1).reset_index(drop=True)
# private_test_data = private_data[int(len(private_data)*0.8):]    
# private_data = private_data[:int(len(private_data)*0.8)]         

concat_data = pd.concat([unsmile_data,private_data], axis=0)
concat_data = concat_data.fillna(0.0)
concat_data = concat_data.reset_index()

concat_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설",'clean','name', 'number', 'address', 'bank', 'person']

concat_num_labels = len(concat_labels) # 15

concat_data_labels_df = concat_data[concat_labels]
concat_data_labels_df = concat_data_labels_df.astype(float)

concat_label = []

for i in range(len(concat_data["여성/가족"])):
    concat_label.append([np.array(concat_data_labels_df.iloc[i], dtype=np.float64)])

concat_label_data = pd.DataFrame(concat_label, columns = ['label'])

concat_data.rename(columns={'문장':'sentence'}, inplace=True)

concat_dataset = pd.concat([concat_data['sentence'], concat_label_data['label']], axis=1)
concat_dataset=concat_dataset.sample(frac=1).reset_index(drop=True)
# concat_dataset.to_csv('data/concat_dataset.csv', sep='\t', index = False)

model_name = 'beomi/KcELECTRA-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
ckpt_name = 'model_save/Hate_Speach-beomi-kcbert-base-10_private_shuffle.pt'
ckpt_name = "model_save/Hate_Speach-beomi-KcELECTRA-base-11_private_shuffle.pt"
model = ElectraForSequenceClassification.from_pretrained(
    model_name,
    num_labels=concat_num_labels,
    problem_type="multi_label_classification"
).cuda()

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.config.id2label = {i: label for i, label in zip(range(concat_num_labels), concat_labels)}
model.config.label2id = {label: i for i, label in zip(range(concat_num_labels), concat_labels)}

test_text, test_labels = (
    concat_dataset["sentence"].values,
    concat_dataset["label"].values,
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(test_text, test_labels)
]

test_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

with torch.no_grad():
    model.eval() 
    for data in tqdm(test_loader):
        text, label = data["data"], data["label"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()

        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        classification_results = output.logits

print({'test': label_ranking_average_precision_score(label.detach().cpu().numpy(), classification_results.detach().cpu().numpy())})