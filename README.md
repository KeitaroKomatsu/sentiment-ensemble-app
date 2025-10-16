# 🧠 日本語感情分析アプリ（Sentiment Ensemble App）

## 📘 概要（Overview）

このプロジェクトは、**日本語テキストの感情分析**を行うWebアプリケーションです。
Logistic Regression、LightGBM、PyTorch（NN）の、三種類のモデルを学習させ、
それらの予測結果を**多数決で統合（アンサンブル）**して最終的な感情を判定します。

Streamlitを用いたシンプルなUIから文章を入力し、
「ポジティブ」か「ネガティブ」かを瞬時に判定できます。

---

## ✨ 特徴（Features）

* 🧩 **3種類の異なるモデルを組み合わせたアンサンブル構成**

  * Logistic Regression（軽量・高精度）
  * LightGBM（決定木ベースの高性能モデル）
  * PyTorch NN（ニューラルネットによる表現学習）

* 📊 **Tfidf + MeCab による日本語前処理**

  * 形態素解析を用いて日本語を単語単位に分割し特徴ベクトル化。

* 💬 **StreamlitによるWebアプリ化**

  * ローカ環境で動作するインタラクティブなUIを構築。

* 🧠 **日本語データセットを使用**

  * Hugging Face上の公開データセット（今回は`sepidmnorozy/Japanese_sentiment`）を活用。

---

## 📂 ディレクトリ構成（Project Structure）

```
sentiment_app/
├── data/                    # 学習データ（CSV）
│   ├── sentiments_train.csv # 学習用データ
│   └── sentiments_val.csv　 # 検証用データ　
├── models/
│   ├── model_lr.py          # Logistic Regression モデル学習、推論
│   ├── model_lgbm.py        # LightGBM モデル学習、推論
│   ├── model_nn.py          # PyTorch モデル学習、推論
│   └── __init__.py
├── ensemble_predict.py      # 各モデルの予測を統合（多数決）
├── app.py                   # Streamlitアプリ本体
├── load_data.py             # データセットの取得・保存スクリプト
├── requirements.txt         # 依存ライブラリ
└── README.md                # プロジェクト説明（本ファイル）
```

---

## ⚙️ 使用技術（Tech Stack）

| 分類     | 技術                                        |
| ------ | ----------------------------------------- |
| 言語     | Python 3.12                               |
| 自然言語処理 | MeCab, TfidfVectorizer |
| モデル学習  | scikit-learn, LightGBM, PyTorch           |
| Webアプリ | Streamlit                                 |
| データセット | Hugging Face - Japanese Sentiment Dataset |
| モデル統合  | アンサンブル（多数決）                               |

---

## 🧰 環境構築（Setup）

### ① 仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate   # mac / linux
# or
venv\Scripts\activate      # windows
```

### ② 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### ③ MeCab のセットアップ

**macOS / Linux**

```bash
sudo apt install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

**Windows**

1. [MeCab公式](https://taku910.github.io/mecab/) からインストーラーをDL
2. 環境変数PATHにMeCabのパスを追加
3. `pip install mecab-python3`

---

## 💾 データセットの取得

Hugging Face から日本語感情分析データをロードしてCSV化します。

```bash
python load_data.py
```

生成されるファイル：

```
data/sentiments_train.csv
data/sentiments_val.csv
```

---

## 🧠 モデル学習

3種類のモデルをそれぞれ学習します。

```bash
python models/model_lr.py
python models/model_lgbm.py
python models/model_nn.py
```

学習済みモデルは `models/` 以下に保存されます（.pkl）。

---

## 🧩 アンサンブル推論

全モデルの結果を統合して感情を判定します。

```bash
python ensemble_predict.py
```

---

## 🌐 Webアプリ実行

Streamlitアプリを起動します。

```bash
streamlit run app.py
```

ブラウザで自動的に開きます。
入力ボックスに文章を入れると、「ポジティブ or ネガティブ」を返します。

---

## 🧮 アンサンブルの仕組み（Voting）

1. 各モデルが文章の感情を独立に判定。
2. 3つの結果を多数決でまとめ、最終判定を決定。

   * 今回はポジティブ:1, ネガティブ: 0となっている
   * 例1: [1, 1, 0] -> 1(ポジティブ)
   * 例2: [0, 0, 0] -> 0(ネガティブ)

---

## 📈 モデル性能

| モデル                 | 精度（Accuracy） |
| ------------------- | ------------ | 
| Logistic Regression | 0.92         | 
| LightGBM            | 0.96         | 
| PyTorch NN          | 0.96         | 

## 使用モデルについて
まず線形モデルであるLogistic Regressionをベースラインとして用い、
非線形性を考慮できるLightGBMで性能を拡張しました。
さらに、PyTorchによるニューラルネットワークを導入し、
より複雑な関係性をモデル化できるかを検証しました。
結果としてLightGBMおよびNNで精度が向上し、
データの非線形構造を捉えられていることが確認できました。
---

## 🧩 今後の改善案（Future Work）

* [ ] データ拡張（SNS・レビューコーパス、あるいは自作データの追加）
* [ ] Transformerモデル（BERTなど）による精度向上
* [ ] 各モデルのハイパーパラメータチューニング
* [ ] データの正規化等の前処理
* [ ] 結果の可視化（グラフ出力）

---

## 🎓 学べること（What I Learned）

* 日本語形態素解析（MeCab）の実践的利用方法
* scikit-learn, LightGBM, PyTorch の使い分け
* アンサンブル手法（多数決）の設計と実装
* StreamlitによるAIモデルのWeb化
* NLPワークフロー全体（前処理(Mecab + VfidfVectorizer) → 学習 → 推論 → UI）

---

## 👤 開発者

**Keitaro Komatsu**

* Email: [kin29sasuke@outlook.com]
* GitHub: [github.com/KeitaroKomatsu](#)
* Keywords: `NLP`, `Machine Learning`, `Streamlit`, `AI Engineer`

---

## ⚠️ 注意事項
このリポジトリは個人の学習目的で作成されたものです。  
利用しているデータセット・ライブラリはすべてオープンソースであり、  
所属組織や第三者とは無関係です。  
---

## 🖼️ スクリーンショット例

![alt text](<スクリーンショット (19).png>)
```
