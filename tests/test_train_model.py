import pandas as pd

from app.ml.Risk_model import TARGET_COLUMN, load_dataset, merge_target_classes


def test_merge_target_classes_collapses_0_and_1_into_low_risk():
    df = pd.DataFrame({
        TARGET_COLUMN: [0.0, 1.0, 2.0, 2.0, 0.0],
        "BMI": [25, 30, 28, 22, 27],
    })

    merged = merge_target_classes(df)

    assert list(merged[TARGET_COLUMN]) == [0, 0, 1, 1, 0]
    assert merged[TARGET_COLUMN].dtype == int


def test_merge_target_classes_does_not_mutate_input():
    df = pd.DataFrame({TARGET_COLUMN: [0.0, 1.0, 2.0]})

    merge_target_classes(df)

    # the original DataFrame's target column should be untouched
    assert list(df[TARGET_COLUMN]) == [0.0, 1.0, 2.0]


def test_load_dataset_drops_rows_with_missing_target(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        f"{TARGET_COLUMN},BMI\n"
        "0.0,25\n"
        ",30\n"      # missing target -> should be dropped
        "2.0,28\n"
    )

    df = load_dataset(csv_path)

    assert len(df) == 2
    assert df[TARGET_COLUMN].isna().sum() == 0