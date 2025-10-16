# prepare_data.py
import pandas as pd
from datasets import load_dataset

# 日本語感情分析データセットを読み込み
dataset = load_dataset("sepidmnorozy/Japanese_sentiment")

# データの中身を確認
print(dataset)
print(dataset["train"][0])

# CSVに保存
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])

train_df.to_csv("data/sentiments_train.csv", index=False)
val_df.to_csv("data/sentiments_val.csv", index=False)

print("✅ データ保存完了！ data/sentiments_train.csv と data/sentiments_val.csv を確認してください。")
print(train_df["label"][9830])
print(train_df["text"][9830])
# negatives = train_df[train_df["label"] == 0]
