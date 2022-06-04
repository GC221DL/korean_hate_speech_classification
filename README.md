# korean_hate_speech_classification

KcBERTm KcElectra를 base로 https://github.com/smilegate-ai/korean_unsmile_dataset 의 데이터와 개인정보를 임의로 생성한 데이터로 finetuning 진행했습니다.  

모델 save 파일

[KcBERT Model](https://huggingface.co/momo/KcBERT-base_Hate_speech_Privacy_Detection)

[KcElectra Model](https://huggingface.co/momo/KcELECTRA-base_Hate_speech_Privacy_Detection)


총 15개의 label을 분류합니다.

["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean", 'name', 'number', 'address', 'bank', 'person']

Hate speach 9개

Privacy 5개 (이름, 전화번호, 주소, 계좌번호, 주민번호)  -> Real data가 아니라 임의로 생성한 데이터 입니다. 

-------
## Reference
- [korean_unsmile_dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)
- [KcBERT](https://github.com/Beomi/KcBERT)
- [KcElectra](https://github.com/Beomi/KcELECTRA)