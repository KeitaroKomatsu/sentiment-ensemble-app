# models/model_lr.py
import joblib
import MeCab
import pandas as pd
import unidic_lite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

TRAIN = "data/sentiments_train.csv"
TEST = "data/sentiments_val.csv"


def mecab_tokenizer(text):
    tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")  # unidic-lite対応
    tagger.parse("")  # 文字化け対策（初期化）
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        word = node.surface
        if word:  # 空白を除外
            tokens.append(word)
        node = node.next
    return tokens


def train_lr(train_csv=TRAIN, test_csv=TEST):

    df_train = pd.read_csv(train_csv)
    X_train, y_train = df_train["text"], df_train["label"]
    df_test = pd.read_csv(test_csv)
    X_val, y_val = df_test["text"], df_test["label"]

    vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer, token_pattern=None, max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_val_vec)
    print("Logistic Regression accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    joblib.dump(model, "models/lr_model.pkl")
    joblib.dump(vectorizer, "models/lr_vectorizer.pkl")
    print("✅ Logistic Regression model saved!")


def predict_lr(texts):
    vectorizer = joblib.load("models/lr_vectorizer.pkl")
    model = joblib.load("models/lr_model.pkl")
    X_vec = vectorizer.transform(texts)
    return model.predict(X_vec)


# ======== メイン ========
if __name__ == "__main__":
    train_lr()
    # 確認用
    sample = ["今日はとても楽しかった！", "牛乳を買い忘れてしまった。"]
    preds = predict_lr(sample)
    print("Prediction:", preds)
