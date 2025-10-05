import pandas as pd
import numpy as np
import os

from imputation.MICE import MICE
from plotting.utils import md_pattern_like, plot_missing_data_pattern
from plotting.diagnostics import stripplot, bwplot, densityplot, xyplot


def main():
    """Run a complete MICE workflow on the NHANES dataset."""

    os.makedirs('additionalz', exist_ok=True)

    df = pd.read_csv("data/nhanes.csv", na_values="NA")
    print("Loaded NHANES dataset with shape:", df.shape)

    pattern_df = md_pattern_like(df)
    print("\nMissing-data pattern (similar to R's md.pattern):\n", pattern_df)
    plot_missing_data_pattern(
        pattern_df, 
        title="NHANES Missing-Data Pattern",
        save_path='additionalz/missing_data_pattern.png'
    )

    cols = df.columns
    predictor_matrix = pd.DataFrame(1, index=cols, columns=cols)
    np.fill_diagonal(predictor_matrix.values, 0)

    mice_imp = MICE(df)
    mice_imp.impute(n_imputations=8, predictor_matrix=predictor_matrix, method="cart")
    imputed_datasets = mice_imp.imputed_datasets

    cols_with_missing = [col for col in df.columns if df[col].isna().any()]
    mice_imp.plot_chain_stats(columns=cols_with_missing)

    print(f"\nGenerated {len(imputed_datasets)} imputed datasets.")

    pooled_summary = mice_imp.pool(summ=True)
    if pooled_summary is not None:
        print("\nPooled estimates (Rubin's rules):\n")
        print(pooled_summary)

    missing_pattern = df.notna().astype(int)

    stripplot(
        imputed_datasets=imputed_datasets,
        missing_pattern=missing_pattern,
        merge_imputations=False,
        save_path='additionalz/stripplot.png'
    )

    bwplot(
        imputed_datasets=imputed_datasets,
        missing_pattern=missing_pattern,
        merge_imputations=False,
        save_path='additionalz/bwplot.png'
    )

    densityplot(
        imputed_datasets=imputed_datasets,
        missing_pattern=missing_pattern,
        save_path='additionalz/densityplot.png'
    )

    xyplot(
        imputed_datasets=imputed_datasets,
        missing_pattern=missing_pattern,
        x="age",
        y="bmi",
        save_path='additionalz/xyplot_age_bmi.png'
    )


if __name__ == "__main__":
    main()