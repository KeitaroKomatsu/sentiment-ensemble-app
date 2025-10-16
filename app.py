# app.py
import streamlit as st

from ensemble_predict import ensemble_predict
from models.model_lr import mecab_tokenizer  # noqa: F401

st.title("🧠 日本語 感情分析アプリ（多数決モデル）")
st.markdown("Logistic Regression + LightGBM + PyTorch による感情分類")

# テキスト入力欄
text = st.text_area("テキストを入力してください", height=150)

if st.button("分析する"):
    if not text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("分析中..."):
            result = ensemble_predict([text])[0]

        # 表示部分
        if result == 1:
            st.success("🌞 ポジティブな感情です！")
        elif result == -1:
            st.error("☁️ ネガティブな感情です。")

st.markdown("---")
st.caption("Powered by scikit-learn + LightGBM + PyTorch + MeCab")
