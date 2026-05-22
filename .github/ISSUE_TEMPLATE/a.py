from datasets import load_dataset

data = load_dataset('jeanlee/kmhas_korean_hate_speech')

train = data['train']       # 78,977건
valid = data['validation']  # 8,776건
test  = data['test']        # 21,939건

# pandas DataFrame으로 변환
df_train = data['train'].to_pandas()
