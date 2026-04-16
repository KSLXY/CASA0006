import pandas as pd

from src.data_pipeline import prepare_dataset


def test_prepare_dataset_filters_invalid_target_and_creates_xy():
    df = pd.DataFrame(
        {
            "accident_severity": [1, 2, 3, -10],
            "number_of_vehicles": [1, 2, 3, 4],
            "number_of_casualties": [1, 2, 2, 3],
            "cloud_cover": [3, 4, 5, 6],
            "sunshine": [5, 4, 3, 2],
            "global_radiation": [110, 120, 130, 140],
            "max_temp": [15, 16, 17, 18],
            "mean_temp": [10, 11, 12, 13],
            "min_temp": [5, 6, 7, 8],
            "precipitation": [0.1, 0.2, 0.3, 0.4],
            "pressure": [1010, 1008, 1005, 1002],
            "snow_depth": [0, 0, 1, 2],
        }
    )
    features = [
        "number_of_vehicles",
        "number_of_casualties",
        "cloud_cover",
        "sunshine",
        "global_radiation",
        "max_temp",
        "mean_temp",
        "min_temp",
        "precipitation",
        "pressure",
        "snow",
    ]
    prepared = prepare_dataset(df, feature_columns=features, target_column="accident_severity")

    assert prepared.removed_invalid_target == 1
    assert len(prepared.X) == 3
    assert len(prepared.y) == 3
    assert set(prepared.y.unique()) == {0, 1, 2}
    # STATS19 semantic mapping check: 1->Fatal(0), 2->Serious(1), 3->Slight(2)
    assert prepared.y.iloc[0] == 0
    assert prepared.y.iloc[1] == 1
    assert prepared.y.iloc[2] == 2
