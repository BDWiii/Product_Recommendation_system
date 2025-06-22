import json, pickle
import numpy as np
from lightfm.data import Dataset


def load_json(path):
    with open(path) as f:
        return json.load(f)


def encode_user(user, scaler):
    age = user.get("age", 25)
    age_norm = scaler.transform([[age]])[0][0]
    return [
        f"age_bin:{int((age_norm + 3) * 2)}",
        f"gender:{user.get('gender', 'unknown')}",
        f"location:{user.get('location', 'unknown')}",
    ]


def recommend(model_path, user_path, items_path, top_k=6):
    user = load_json(user_path)
    items = load_json(items_path)

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    dataset = data["dataset"]
    scaler = data["scaler"]
    item_feats = data["item_feats"]
    item_mat = data["item_features"]

    user_feats = encode_user(user, scaler)
    user_mat = dataset.build_user_features(
        [(user["user_id"], user_feats)], normalize=False
    )

    item_ids = [i["item_id"] for i in items if i["item_id"] in item_feats]
    item_indices = [dataset.mapping()[2][item_id] for item_id in item_ids]

    scores = model.predict(
        0, item_ids=item_indices, user_features=user_mat, item_features=item_mat
    )
    ranked = sorted(zip(item_ids, scores), key=lambda x: -x[1])[:top_k]

    return [{"item_id": iid, "score": float(score)} for iid, score in ranked]


if __name__ == "__main__":
    recs = recommend("model.pkl", "user_to_predict.json", "items_for_prediction.json")
    for r in recs:
        print(f"{r['item_id']} -> {r['score']:.4f}")
