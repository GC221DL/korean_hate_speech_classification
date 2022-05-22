import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from torch.optim import AdamW
from kobert_tokenizer import KoBERTTokenizer

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

model = BertForSequenceClassification.from_pretrained(
    'skt/kobert-base-v1',
    num_labels=10,
).cuda()

train_data = pd.read_csv("data/train_data.csv", delimiter="\t")
train_data = train_data[:int(len(train_data)*0.8)]
eval_data = train_data[int(len(train_data)*0.8):]

train_text, train_labels = (
    train_data["sentence"].values,
    train_data["label"].values,
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(train_text, train_labels)
]

train_loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

# eval_data = pd.read_csv("data/train_data.csv", delimiter="\t")
# eval_data = eval_data[100000:]
eval_text, eval_labels = (
    eval_data["sentence"].values,
    eval_data["label"].values
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = AdamW(params=model.parameters(),
    lr=3e-5, weight_decay=3e-7
)

epochs = 1
for epoch in range(epochs):
    model.train()
    for train in tqdm(train_loader):
        optimizer.zero_grad()
        text, label = train["data"], train["label"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            # max_length=140

        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )

        loss = output.loss
        loss.backward()        
        optimizer.step()
        classification_results = output.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

    print({"loss": loss.item()})
    print({"acc": acc / len(classification_results)})   ## 탭하나 안에 넣으면 step단위로 볼수있음. 

    with torch.no_grad():
        model.eval() 
        for eval in tqdm(eval_loader):
            eval_text, eval_label = eval["data"], eval["label"]
            eval_tokens = tokenizer(
                eval_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                # max_length=140
            )
            input_ids = eval_tokens.input_ids.cuda()
            attention_mask = eval_tokens.attention_mask.cuda()

            eval_out = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=eval_label
            )

            eval_classification_results = eval_out.logits.argmax(-1)
            eval_loss = eval_out.loss

            eval_acc = 0
            for res, lab in zip(eval_classification_results, eval_label):
                if res == lab:
                    eval_acc += 1

        print({"eval_loss": eval_loss.item()})   ## 이미 다 적용된 상태인듯..
        print({"eval_acc": eval_acc / len(eval_classification_results)})             ## 탭하나 안에 넣으면 step단위로 볼수있음. 
        print({"epoch": epoch+1})
        torch.save(model.state_dict(), f"model_save/BERT_fintuing_NSMC-{epoch+1}.pt")