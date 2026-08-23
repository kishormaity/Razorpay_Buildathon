import os
import pandas as pd

def main():
    print("=" * 70)
    print("              TRANSACTION FEATURES VERIFICATION RUN")
    print("=" * 70)
    
    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    path = os.path.join(current_dir, "data", "processed", "features/transaction_features.parquet")
    
    if not os.path.exists(path):
        print(f"[ERROR] Features file not found at: {path}")
        return

    df = pd.read_parquet(path)

    print("Shape:", df.shape)

    print("\nDtypes Count:")
    print(df.dtypes.value_counts())

    print("\nTarget Class Distribution:")
    print(df["isFraud"].value_counts())
    
    fraud_rate = (df["isFraud"].sum() / len(df)) * 100
    print(f"Calculated Fraud Rate: {fraud_rate:.3f}%")

    print("\nMissingness flags null counts:")
    flags = [
        "is_card_missing",
        "is_email_missing",
        "is_address_missing",
        "is_device_missing",
        "is_identity_missing"
    ]
    print(df[flags].isna().sum())

    print("\nDerived features sample (head 10):")
    print(
        df[
            [
                "log_transaction_amount",
                "card_type_combination",
                "email_domain_match"
            ]
        ].head(10)
    )

    print("\nCategorical columns:")
    categorical_cols = df.select_dtypes(
        include=["category"]
    ).columns
    print(list(categorical_cols))
    
    print("\n" + "=" * 70)
    print("Verification execution complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
