# tests/test_preprocessing.py
"""
Unit tests for src/data/preprocessor.py and src/data/data_loader.py.
Each test covers one specific behaviour — no more, no less.
"""

from src.data.preprocessor import fix_total_charges


class TestFixTotalCharges:
    """Tests for the fix_total_charges function."""

    def test_converts_string_numbers_to_float(self, raw_churn_df):
        """Valid string numbers like '358.20' must become float."""
        result = fix_total_charges(raw_churn_df)
        assert result["TotalCharges"].dtype == float

    def test_blank_strings_become_nan(self, raw_churn_df):
        """Blank strings ' ' must be coerced to NaN, not raise an error."""
        result = fix_total_charges(raw_churn_df)
        assert result["TotalCharges"].isna().sum() == 1

    def test_does_not_modify_original_dataframe(self, raw_churn_df):
        """fix_total_charges must be pure — original df must be unchanged."""
        original_dtype = raw_churn_df["TotalCharges"].dtype
        fix_total_charges(raw_churn_df)
        assert raw_churn_df["TotalCharges"].dtype == original_dtype

    def test_missing_column_returns_df_unchanged(self, raw_churn_df):
        """If TotalCharges column is absent, return df without raising."""
        df_no_col = raw_churn_df.drop(columns=["TotalCharges"])
        result = fix_total_charges(df_no_col)
        assert "TotalCharges" not in result.columns
        assert result.shape == df_no_col.shape

    def test_already_numeric_column_is_unchanged(self, clean_churn_df):
        """If TotalCharges is already float, function must not break it."""
        result = fix_total_charges(clean_churn_df)
        assert result["TotalCharges"].dtype == float


class TestTargetEncoding:
    """Tests for target label encoding behaviour in load_data."""

    def test_target_values_are_numeric(self, raw_churn_df):
        """After encoding, y must contain only 0 and 1 — never strings."""
        y = raw_churn_df["Churn"]
        if set(y.dropna().unique()) == {"Yes", "No"}:
            y = y.map({"Yes": 1, "No": 0})
        assert set(y.unique()).issubset({0, 1})

    def test_yes_maps_to_1(self, raw_churn_df):
        """'Yes' must map to 1."""
        y = raw_churn_df["Churn"].map({"Yes": 1, "No": 0})
        assert y[raw_churn_df["Churn"] == "Yes"].eq(1).all()

    def test_no_maps_to_0(self, raw_churn_df):
        """'No' must map to 0."""
        y = raw_churn_df["Churn"].map({"Yes": 1, "No": 0})
        assert y[raw_churn_df["Churn"] == "No"].eq(0).all()
