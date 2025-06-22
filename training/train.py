import json, pickle
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing import StandardScaler

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_features(users, items):
    genders = {u.get('gender', 'unknown') for u in users}
    locations = {u.get('location', 'unknown') for u in users}
    categories = {i.get('category', 'unknown') for i in items}
    brands = {i.get('brand', 'unknown') for i in items}
    prices = {i.get('price', 'unknown') for i in items}
    return genders, locations, categories, brands, prices

def encode_user_features(user, scaler):
    age = user.get('age', 25)
    age_norm = scaler.transform([[age]])[0][0]
    return [
        f"age_bin:{int((age_norm + 3) * 2)}",
        f"gender:{user.get('gender', 'unknown')}",
        f"location:{user.get('location', 'unknown')}"
    ]

def encode_item_features(item):
    return [
        f"category:{item.get('category', 'unknown')}",
        f"price:{item.get('price', 'unknown')}",
        f"brand:{item.get('brand', 'unknown')}"
    ]

def train(users_path, items_path, inters_path, output='model.pkl'):
    users = load_json(users_path)
    items = load_json(items_path)
    inters = load_json(inters_path)

    scaler = StandardScaler()
    scaler.fit(np.array([u.get('age', 25) for u in users]).reshape(-1, 1))

    genders, locs, cats, brands, prices = extract_features(users, items)

    user_ids = [u['user_id'] for u in users]
    item_ids = [i['item_id'] for i in items]

    user_feats = {u['user_id']: encode_user_features(u, scaler) for u in users}
    item_feats = {i['item_id']: encode_item_features(i) for i in items}

    dataset = Dataset()
    dataset.fit(user_ids, item_ids,
                user_features=set(f for fs in user_feats.values() for f in fs),
                item_features=set(f for fs in item_feats.values() for f in fs))

    inter_mat, _ = dataset.build_interactions([
        (i['user_id'], i['item_id'], i.get('rating', 1.0)) for i in inters
    ])
    user_mat = dataset.build_user_features(user_feats.items())
    item_mat = dataset.build_item_features(item_feats.items())

    model = LightFM(loss='warp')
    model.fit(inter_mat, user_features=user_mat, item_features=item_mat, epochs=30, num_threads=2)

    with open(output, 'wb') as f:
        pickle.dump({
            'model': model,
            'dataset': dataset,
            'user_feats': user_feats,
            'item_feats': item_feats,
            'scaler': scaler
        }, f)

    print(f"✓ Trained model saved to {output}")

if __name__ == "__main__":
    train('users.json', 'items.json', 'interactions.json')
