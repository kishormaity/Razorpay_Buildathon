import os
import sys
import time
import json


def main():
    print("=" * 70)
    print("          IEEE-CIS DATASET MERGE & PARQUET BUILDER")
    print("=" * 70)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    raw_dir = os.path.join(
        current_dir,
        "data",
        "raw"
    )

    processed_dir = os.path.join(
        current_dir,
        "data",
        "processed"
    )

    txn_path = os.path.join(
        raw_dir,
        "train_transaction.csv"
    )

    id_path = os.path.join(
        raw_dir,
        "train_identity.csv"
    )

    output_path = os.path.join(
        processed_dir,
        "merged_train.parquet"
    )

    metadata_path = os.path.join(
        processed_dir,
        "merge_metadata.json"
    )

    # ---------------------------------------------------------
    # 1. Validate files
    # ---------------------------------------------------------

    print("\n[1/6] Validating input files...")

    if not os.path.exists(txn_path):
        print(f"[ERROR] Missing: {txn_path}")
        sys.exit(1)

    if not os.path.exists(id_path):
        print(f"[ERROR] Missing: {id_path}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Dependencies
    # ---------------------------------------------------------

    try:
        import pandas as pd
        import pyarrow
    except ImportError:
        print(
            "[ERROR] Required packages missing.\n"
            "Run:\n"
            "pip install pandas pyarrow"
        )
        sys.exit(1)

    os.makedirs(processed_dir, exist_ok=True)

    start_time = time.time()

    # ---------------------------------------------------------
    # 3. Load datasets
    # ---------------------------------------------------------

    print("\n[2/6] Loading identity dataset...")

    t0 = time.time()

    id_df = pd.read_csv(id_path)

    print(
        f"Identity shape: {id_df.shape}"
        f" ({time.time() - t0:.2f}s)"
    )

    print("\n[3/6] Loading transaction dataset...")

    t0 = time.time()

    tx_df = pd.read_csv(txn_path)

    print(
        f"Transaction shape: {tx_df.shape}"
        f" ({time.time() - t0:.2f}s)"
    )

    # ---------------------------------------------------------
    # 4. Validate keys
    # ---------------------------------------------------------

    print("\n[4/6] Validating TransactionID keys...")

    if "TransactionID" not in tx_df.columns:
        raise ValueError(
            "TransactionID missing from transaction dataset."
        )

    if "TransactionID" not in id_df.columns:
        raise ValueError(
            "TransactionID missing from identity dataset."
        )

    tx_duplicates = tx_df["TransactionID"].duplicated().sum()
    id_duplicates = id_df["TransactionID"].duplicated().sum()

    print(f"Transaction duplicate IDs: {tx_duplicates}")
    print(f"Identity duplicate IDs: {id_duplicates}")

    if tx_duplicates > 0:
        raise ValueError(
            "Duplicate TransactionIDs found in transaction dataset."
        )

    if id_duplicates > 0:
        raise ValueError(
            "Duplicate TransactionIDs found in identity dataset."
        )

    # ---------------------------------------------------------
    # 5. Merge
    # ---------------------------------------------------------

    print("\n[5/6] Merging datasets...")

    t0 = time.time()

    original_rows = len(tx_df)

    merged_df = pd.merge(
        tx_df,
        id_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    merge_time = time.time() - t0

    print(
        f"Merged shape: {merged_df.shape}"
        f" ({merge_time:.2f}s)"
    )

    # ---------------------------------------------------------
    # Merge validation
    # ---------------------------------------------------------

    if len(merged_df) != original_rows:
        raise ValueError(
            "Row count changed after merge!"
        )

    unmatched_identity = (
        merged_df["_merge"] == "left_only"
    ).sum()

    matched_identity = (
        merged_df["_merge"] == "both"
    ).sum()

    identity_match_rate = (
        matched_identity / original_rows
    ) * 100

    print(
        f"Identity matches: {matched_identity:,}"
    )

    print(
        f"Identity match rate: "
        f"{identity_match_rate:.2f}%"
    )

    print(
        f"Transactions without identity: "
        f"{unmatched_identity:,}"
    )

    # Remove merge helper column
    merged_df.drop(
        columns=["_merge"],
        inplace=True
    )

    # ---------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------

    fraud_count = int(
        merged_df["isFraud"].sum()
    )

    fraud_rate = (
        merged_df["isFraud"].mean() * 100
    )

    print(
        f"Fraud cases: {fraud_count:,}"
    )

    print(
        f"Fraud rate: {fraud_rate:.3f}%"
    )

    # ---------------------------------------------------------
    # 6. Write Parquet
    # ---------------------------------------------------------

    print("\n[6/6] Writing Parquet...")

    t0 = time.time()

    merged_df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False
    )

    write_time = time.time() - t0

    file_size_mb = (
        os.path.getsize(output_path)
        / (1024 * 1024)
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "source_transaction_file": txn_path,
        "source_identity_file": id_path,
        "output_file": output_path,
        "transaction_rows": int(len(tx_df)),
        "identity_rows": int(len(id_df)),
        "merged_rows": int(len(merged_df)),
        "merged_columns": int(len(merged_df.columns)),
        "identity_matches": int(matched_identity),
        "identity_match_rate_pct": round(
            identity_match_rate,
            3
        ),
        "fraud_cases": fraud_count,
        "fraud_rate_pct": round(
            fraud_rate,
            3
        ),
        "parquet_size_mb": round(
            file_size_mb,
            2
        ),
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"Rows:             {len(merged_df):,}")
    print(f"Columns:          {len(merged_df.columns):,}")
    print(f"Fraud cases:      {fraud_count:,}")
    print(f"Fraud rate:       {fraud_rate:.3f}%")
    print(f"Identity match:   {identity_match_rate:.2f}%")
    print(f"Parquet size:     {file_size_mb:.2f} MB")
    print(f"Total time:       {elapsed:.2f}s")

    print(f"\nDataset:")
    print(output_path)

    print(f"\nMetadata:")
    print(metadata_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
