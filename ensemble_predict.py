# ensemble_predict.py
import numpy as np

from models.model_lgbm import predict_lgbm
from models.model_lr import predict_lr
from models.model_pytorch import predict_nn


def ensemble_predict(texts):
    """
    3モデル（LR, LightGBM, NN）の結果を多数決で統合する
    """
    preds_lr = predict_lr(texts)
    preds_lgbm = predict_lgbm(texts)
    preds_nn = predict_nn(texts)

    preds = np.array([preds_lr, preds_lgbm, preds_nn])  # shape (3, n_samples)
    final = []

    for i in range(preds.shape[1]):
        votes = preds[:, i]
        # 多数決
        if votes.sum() >= 2:
            final.append(1)
        else:
            final.append(-1)
    return final


if __name__ == "__main__":
    sample = [
        "今日はとても嬉しい気分です！",
        "この映画は最悪だった。",
        "うーん、まあまあかな。",
    ]
    results = ensemble_predict(sample)
    print("✅ Ensemble Results:", results)
