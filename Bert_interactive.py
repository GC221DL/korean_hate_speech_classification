"""
python interactive.py
"""
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline

model_name = 'beomi/kcbert-base'

ckpt_name = "model_save/Hate_Speach-beomi-kcbert-base-20.pt"

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=10,
    problem_type="multi_label_classification"
).cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]
num_labels = len(unsmile_labels)

model.config.id2label = {i: label for i, label in zip(range(num_labels), unsmile_labels)}
model.config.label2id = {label: i for i, label in zip(range(num_labels), unsmile_labels)}

pipe = TextClassificationPipeline(
    model = model,
    tokenizer = tokenizer,
    device=0,
    return_all_scores=True,
    function_to_apply='sigmoid'
    )
    
sentence = "이래서 여자는 게임을 하면 안된다"
print({"sentence": sentence})
for result in pipe(sentence)[0]:
    print(result)