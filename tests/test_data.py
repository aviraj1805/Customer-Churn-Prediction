import pandas as pd
import pytest

from src.data import encode_target, load_raw, split_X_y, stratified_sample, train_test
from src.features import CATEGORICAL_FEATURES, RAW_FEATURES
from tests.conftest import make_raw_frame


def test_load_raw_reads_all_columns_with_categorical_dtype(raw_csv):
    df = load_raw(raw_csv)
    assert set(RAW_FEATURES) <= set(df.columns)
    assert all(isinstance(df[col].dtype, pd.CategoricalDtype) for col in CATEGORICAL_FEATURES)


def test_load_raw_rejects_file_with_missing_column(tmp_path, raw_df):
    path = tmp_path / "broken.csv"
    raw_df.drop(columns="Contract").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Contract"):
        load_raw(path)


def test_encode_target_maps_yes_to_one():
    y = encode_target(pd.Series(["Yes", "No", "Yes", "No"]))
    assert y.tolist() == [1, 0, 1, 0]


def test_encode_target_rejects_missing_labels():
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", None]))


def test_split_X_y_drops_id_and_target(raw_df, cfg):
    X, y = split_X_y(raw_df, cfg)
    assert list(X.columns) == RAW_FEATURES
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(raw_df)


def test_train_test_is_stratified_and_reproducible(cfg):
    X, y = split_X_y(make_raw_frame(n_rows=2000), cfg)
    X_train, X_test, y_train, y_test = train_test(X, y, cfg)

    assert len(X_test) == pytest.approx(len(X) * cfg["data"]["test_size"], abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)
    assert X_test.index.equals(train_test(X, y, cfg)[1].index)


def test_stratified_sample_keeps_class_balance(cfg):
    X, y = split_X_y(make_raw_frame(n_rows=2000), cfg)
    X_small, y_small = stratified_sample(X, y, n_rows=500, seed=cfg["seed"])
    assert len(X_small) == 500
    assert y_small.mean() == pytest.approx(y.mean(), abs=0.01)
    assert stratified_sample(X, y, n_rows=None, seed=cfg["seed"])[0] is X
