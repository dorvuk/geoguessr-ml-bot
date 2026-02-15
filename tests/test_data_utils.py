import pandas as pd

from geobot.data_utils import assign_group_split


def test_group_split_deterministic() -> None:
    df = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "sequence": ["s1", "s2", "s3", "s4"],
        }
    )
    split_a = assign_group_split(df, group_col="sequence", val_ratio=0.1, test_ratio=0.1, seed=42)
    split_b = assign_group_split(df, group_col="sequence", val_ratio=0.1, test_ratio=0.1, seed=42)
    assert split_a.tolist() == split_b.tolist()
