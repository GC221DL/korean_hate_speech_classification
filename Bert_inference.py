"""
python inference.py
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import label_ranking_average_precision_score

model_name = 'beomi/kcbert-base'

ckpt_name = "model_save/Hate_Speach-beomi-kcbert-base-20.pt"
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=10,
    problem_type="multi_label_classification"
).cuda()

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]
num_labels = len(unsmile_labels)

tokenizer = AutoTokenizer.from_pretrained(model_name)


model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

## data
test_data = pd.read_csv('data/unsmile_valid_v1.0.tsv', delimiter='\t')

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]
num_labels = len(unsmile_labels)

labels_df = test_data[["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]]
labels_df = labels_df.astype(float)

label = []

for i in range(len(test_data["여성/가족"])):

    label.append([np.array(labels_df.iloc[i], dtype=np.float64)])

label_data = pd.DataFrame(label, columns = ['label'])

test_data.rename(columns={'문장':'sentence'}, inplace=True)

dataset = pd.concat([test_data['sentence'], label_data['label']], axis=1)

test_text, test_labels = (
    dataset["sentence"].values,
    dataset["label"].values,
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