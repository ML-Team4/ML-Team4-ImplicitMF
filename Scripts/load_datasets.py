from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
from scipy.sparse import csr_matrix

DATA_DIR = Path("datasets/listenbrainz-preprocessing")

def read(name, types):
    return pc.read_csv(DATA_DIR/ name,
        parse_options=pc.ParseOptions(newlines_in_values=True),
        convert_options=pc.ConvertOptions(column_types=types,
            null_values=[], strings_can_be_null=False))

def load_data() -> dict:
    users = read(
        "users.csv",{"user_idx": pa.int32(), "user_id": pa.int64()},)
    items = read(
        "items.csv",{"item_idx": pa.int32(), "artist_name": pa.string(),"recording_name": pa.string(), "train_listener_count": pa.int32(),"artist_key": pa.string(), "recording_key": pa.string()}, )
    train = read(
        "train_interactions.csv",{"user_idx": pa.int32(),"item_idx": pa.int32(), "listen_count": pa.int64()}, )
    counts = csr_matrix((train["listen_count"].to_numpy(),(train["user_idx"].to_numpy(),train["item_idx"].to_numpy())),shape=(users.num_rows, items.num_rows))

    # 모델이 추천한 공통 item_idx를 곡명·아티스트명으로 조회한다.
    recommended_item_idx = [0, 1, 2]  # 실행 예시 인덱스; 실제 모델의 추천 결과로 교체
    lookup = {row["item_idx"]: row for row in
        items.select(["item_idx", "artist_name", "recording_name", "train_listener_count"])
            .take(pa.array(recommended_item_idx, type=pa.int32())).to_pylist()}
    for item_idx in recommended_item_idx:
        print(item_idx, lookup[item_idx])

    # 평가 사용자와 정답은 각 기간의 파일을 짝지어 사용한다.
    evaluation = {}
    for period in ("validation", "test"):
        eligible = read(f"{period}_users.csv", {"user_idx": pa.int32()})["user_idx"].to_numpy()
        targets = read(f"{period}_targets.csv", {"user_idx": pa.int32(), "item_idx": pa.int32()})
        target_u, target_i = (targets[k].to_numpy() for k in ("user_idx", "item_idx"))
        assert np.array_equal(np.unique(target_u), eligible)
        evaluation[period] = (eligible, targets)
        user_idx = int(eligible[0])
        truth = set(target_i[target_u == user_idx].tolist())
        train_seen = set(counts.getrow(user_idx).indices.tolist())
        assert truth.isdisjoint(train_seen)

    return {
        "users": users,
        "items": items,
        "counts": counts,
        "evaluation": evaluation
    }

if __name__ == "__main__":
    dataset = load_data()