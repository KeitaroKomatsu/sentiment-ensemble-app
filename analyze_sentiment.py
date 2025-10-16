from transformers import pipeline

# 学習済みの英語感情分析モデルをロード
model = "KoichiYasuoka/bert-base-japanese-wikipedia-ud-head"
classifier = pipeline("sentiment-analysis", model=model)

# テスト用テキスト
texts = [
    "今日はとても楽しい気分です！",
    "この映画は本当に退屈だった。",
    "まあまあかな、悪くはないけど特別良くもない。",
]

# 各文の感情を分析
for text in texts:
    result = classifier(text)
    label = result[0]["label"]
    score = result[0]["score"]
    print(f"Text: {text}")
    print(f" → Sentiment: {label} (Confidence: {score:.2f})\n")
