import pandas as pd
import numpy as np
import os

# Import the custom MICE implementation
from imputation.MICE import MICE

# Visualization helpers
from plotting.utils import md_pattern_like, plot_missing_data_pattern
from plotting.diagnostics import stripplot, bwplot, densityplot, xyplot


def main():
    """Run a complete MICE workflow on the NHANES dataset."""

    # Create additionalz directory if it doesn't exist
    os.makedirs('additionalz', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    df = pd.read_csv("data/nhanes.csv", na_values="NA")
    print("Loaded NHANES dataset with shape:", df.shape)

    # ------------------------------------------------------------------
    # 2. Inspect and display the missing-data pattern
    # ------------------------------------------------------------------
    pattern_df = md_pattern_like(df)
    print("\nMissing-data pattern (similar to R's md.pattern):\n", pattern_df)
    plot_missing_data_pattern(
        pattern_df, 
        title="NHANES Missing-Data Pattern",
        save_path='additionalz/missing_data_pattern.png'
    )

    # ------------------------------------------------------------------
    # 3. Build a predictor matrix where every variable predicts every other
    #    (i.e., all 1s except the diagonal).
    # ------------------------------------------------------------------
    cols = df.columns
    predictor_matrix = pd.DataFrame(1, index=cols, columns=cols)
    np.fill_diagonal(predictor_matrix.values, 0)  # A variable should not predict itself

    # ------------------------------------------------------------------
    # 4. Create the MICE imputer and generate 5 imputed datasets
    # ------------------------------------------------------------------
    mice_imp = MICE(df)
    mice_imp.impute(n_imputations=8, predictor_matrix=predictor_matrix, method="cart")
    imputed_datasets = mice_imp.imputed_datasets

    # ------------------------------------------------------------------
    # 4b. Visualize convergence of chain statistics (mean & variance)
    # ------------------------------------------------------------------
    cols_with_missing = [col for col in df.columns if df[col].isna().any()]
    mice_imp.plot_chain_stats(columns=cols_with_missing)

    # print(mice_imp.chain_mean)
    # print(mice_imp.chain_var)

    print(f"\nGenerated {len(imputed_datasets)} imputed datasets.")

    # ------------------------------------------------------------------
    # 5. Pool the results (Rubin's rules) and print a summary
    # ------------------------------------------------------------------
    pooled_summary = mice_imp.pool(summ=True)
    if pooled_summary is not None:
        print("\nPooled estimates (Rubin's rules):\n")
        print(pooled_summary)

    # ------------------------------------------------------------------
    # 6. Prepare a simple 0/1 missing-pattern DataFrame required by the
    #    diagnostic plotting functions (1 = observed, 0 = missing).
    # ------------------------------------------------------------------
    missing_pattern = df.notna().astype(int)

    # ------------------------------------------------------------------
    # 7. Diagnostic plots
    # ------------------------------------------------------------------
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

    # Example XY plot: age (complete) vs bmi (contains missing values)
    xyplot(
        imputed_datasets=imputed_datasets,
        missing_pattern=missing_pattern,
        x="age",
        y="bmi",
        save_path='additionalz/xyplot_age_bmi.png'
    )


if __name__ == "__main__":
    main()