# models/model_pytorch.py
import joblib
import MeCab
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# ======== 日本語形態素解析（MeCab） ========
def mecab_tokenizer(text):
    tagger = MeCab.Tagger("")  # unidic-lite対応
    tagger.parse("")
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        word = node.surface
        if word:
            tokens.append(word)
        node = node.next
    return tokens


# ======== PyTorchモデル定義 ========
class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# ======== 学習関数 ========
def train_nn(
    train_csv="data/sentiments_train.csv",
    test_csv="data/sentiments_val.csv",
    num_epochs=10,
    batch_size=64,
):
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
        ngram_range=(1, 2),
    )
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_val_vec = vectorizer.transform(X_val).toarray()

    joblib.dump(vectorizer, "models/nn_vectorizer.pkl")

    # Tensor変換
    X_train_t = torch.tensor(X_train_vec, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_vec, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    model = SentimentNN(input_dim=X_train_vec.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training PyTorch NN...")
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), batch_size):
            idx = permutation[i : i + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 検証
    model.eval()
    with torch.no_grad():
        preds = (model(X_val_t) > 0.5).int().numpy().flatten()
        acc = accuracy_score(y_val_t, preds)
        print(f"✅ PyTorch NN accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "models/nn_model.pth")
    print("💾 PyTorch model and vectorizer saved successfully to models/")


# ======== 推論関数 ========
def predict_nn(texts):
    vectorizer = joblib.load("models/nn_vectorizer.pkl")
    X_vec = vectorizer.transform(texts).toarray()
    X_t = torch.tensor(X_vec, dtype=torch.float32)

    model = SentimentNN(input_dim=X_vec.shape[1])
    model.load_state_dict(torch.load("models/nn_model.pth"))
    model.eval()

    with torch.no_grad():
        preds = (model(X_t) > 0.5).int().numpy().flatten()
    return preds


# ======== メイン ========
if __name__ == "__main__":
    train_nn()
    # 確認用
    sample = ["今日はとても楽しかった！", "牛乳を買い忘れてしまった。"]
    preds = predict_nn(sample)
    print("Prediction:", preds)
