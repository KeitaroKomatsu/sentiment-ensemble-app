# models/model_lgbm.py
import joblib
import MeCab
import pandas as pd
import unidic_lite
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


# ======== 日本語形態素解析（MeCab） ========
def mecab_tokenizer(text):
    """
    MeCabで日本語文を単語ごとに分割するトークナイザ
    """
    tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")  # unidic-lite
    tagger.parse("")  # 初期化
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        word = node.surface
        if word:
            tokens.append(word)
        node = node.next
    return tokens


# ======== 学習関数 ========
def train_lgbm(train_csv="data/sentiments_train.csv", test_csv="data/sentiments_val.csv"):
    print("📘 Loading dataset...")
    df_train = pd.read_csv(train_csv)
    X_train, y_train = df_train["text"], df_train["label"]
    df_test = pd.read_csv(test_csv)
    X_val, y_val = df_test["text"], df_test["label"]

    print("🔧 Vectorizing text with MeCab + TF-IDF...")
    vectorizer = TfidfVectorizer(
        tokenizer=mecab_tokenizer,
        token_pattern=None,
        max_features=8000,
        ngram_range=(1, 2),  # bigram対応
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print("🚀 Training LightGBM classifier...")
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_val_vec)
    acc = accuracy_score(y_val, preds)
    print(f"✅ LightGBM accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    joblib.dump(model, "models/lgbm_model.pkl")
    joblib.dump(vectorizer, "models/lgbm_vectorizer.pkl")
    print("💾 LightGBM model saved successfully to models/")


# ======== 推論関数 ========
def predict_lgbm(texts):
    vectorizer = joblib.load("models/lgbm_vectorizer.pkl")
    model = joblib.load("models/lgbm_model.pkl")
    X_vec = vectorizer.transform(texts)
    preds = model.predict(X_vec)
    return preds


# ======== メイン ========
if __name__ == "__main__":
    train_lgbm()
    # 確認用
    sample = ["今日はとても楽しかった！", "牛乳を買い忘れてしまった。"]
    preds = predict_lgbm(sample)
    print("Prediction:", preds)
