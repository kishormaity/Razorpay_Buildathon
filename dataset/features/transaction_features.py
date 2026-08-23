import os
import sys
import time
import pandas as pd
import numpy as np

def main():
    print("=" * 70)
    print("        IEEE-CIS TRANSACTION FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    input_path = os.path.join(processed_dir, 'features/merged_train.parquet')
    output_path = os.path.join(processed_dir, 'features/transaction_features.parquet')

    # 1. Validate Input File
    print("\n[1/5] Validating input file...")
    if not os.path.exists(input_path):
        print(f"[ERROR] Merged dataset not found at: {input_path}")
        print("Please run build_merged_dataset.py first.")
        sys.exit(1)

    start_time = time.time()

    # 2. Load Dataset
    print(f"Loading merged dataset from {os.path.basename(input_path)}...")
    t0 = time.time()
    df = pd.read_parquet(input_path)
    print(f"Loaded dataset in {time.time() - t0:.2f} seconds. Shape: {df.shape}")

    # Validate core columns
    required_cols = ['TransactionID', 'TransactionAmt', 'TransactionDT']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # 3. Feature Grouping & Selection
    print("\n[2/5] Selecting base numerical and categorical features...")
    
    # Direct Numerical features (TransactionAmt, TransactionDT, dist1, dist2, C*, D*, V*)
    base_numerical = ['TransactionAmt', 'TransactionDT']
    for col in ['dist1', 'dist2']:
        if col in df.columns:
            base_numerical.append(col)
            
    c_cols = sorted([c for c in df.columns if c.startswith('C') and c[1:].isdigit()])
    d_cols = sorted([c for c in df.columns if c.startswith('D') and c[1:].isdigit()])
    v_cols = sorted([c for c in df.columns if c.startswith('V') and c[1:].isdigit()])
    
    numerical_features = base_numerical + c_cols + d_cols + v_cols
    numerical_features = [c for c in numerical_features if c in df.columns]
    print(f"-> Selected {len(numerical_features)} numerical features.")

    # Direct Categorical features
    target_categoricals = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
    ]
    categorical_features = [c for c in target_categoricals if c in df.columns]
    print(f"-> Selected {len(categorical_features)} categorical features.")

    # 4. Feature Derivation (Null-Safe)
    print("\n[3/5] Computing derived transaction features...")
    
    # log_transaction_amount
    df['log_transaction_amount'] = np.log1p(df['TransactionAmt'])
    
    # card_type_combination (with UNKNOWN mapping for missing values)
    card4_clean = df['card4'].fillna('UNKNOWN').astype(str)
    card6_clean = df['card6'].fillna('UNKNOWN').astype(str)
    df['card_type_combination'] = (card4_clean + "_" + card6_clean).astype('category')

    # email_domain_match: 1.0 (match), 0.0 (mismatch), NaN (either missing)
    email_match = pd.Series(np.nan, index=df.index)
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        p_email = df['P_emaildomain']
        r_email = df['R_emaildomain']
        valid_mask = p_email.notna() & r_email.notna()
        email_match[valid_mask] = (p_email[valid_mask] == r_email[valid_mask]).astype(float)
    df['email_domain_match'] = email_match

    # 5. Missingness Flags
    print("\n[4/5] Building missingness indicator flags...")
    
    df['is_card_missing'] = df['card1'].isna().astype('int8') if 'card1' in df.columns else np.int8(1)
    df['is_email_missing'] = df['P_emaildomain'].isna().astype('int8') if 'P_emaildomain' in df.columns else np.int8(1)
    df['is_address_missing'] = df['addr1'].isna().astype('int8') if 'addr1' in df.columns else np.int8(1)
    df['is_device_missing'] = df['DeviceInfo'].isna().astype('int8') if 'DeviceInfo' in df.columns else np.int8(1)
    
    # is_identity_missing (True only if ALL identity features are null)
    identity_cols = [col for col in df.columns if col == 'DeviceType' or col == 'DeviceInfo' or (col.startswith('id_') and col[3:].isdigit())]
    if identity_cols:
        df['is_identity_missing'] = df[identity_cols].isna().all(axis=1).astype('int8')
    else:
        df['is_identity_missing'] = pd.Series(1, index=df.index, dtype='int8')

    # Cast IDs and codes to string then categorical type so LightGBM handles them correctly
    print("Casting categorical columns to category data type...")
    for col in categorical_features:
        df[col] = df[col].astype('string').astype('category')

    # 6. Save Engine Output
    print("\n[5/5] Writing transaction features Parquet...")
    
    # Explicit separation of metadata/target from inputs
    metadata_cols = ['TransactionID']
    target_cols = ['isFraud'] if 'isFraud' in df.columns else []
    derived_cols = ['log_transaction_amount', 'card_type_combination', 'email_domain_match']
    missing_flags = ['is_card_missing', 'is_email_missing', 'is_address_missing', 'is_device_missing', 'is_identity_missing']
    
    final_cols = metadata_cols + target_cols + numerical_features + categorical_features + derived_cols + missing_flags
    # Retain only exists
    final_cols = [c for c in final_cols if c in df.columns]
    
    feature_df = df[final_cols]
    
    t0 = time.time()
    feature_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Successfully saved transaction features to: {output_path}")
    print(f"Parquet write time: {time.time() - t0:.2f} seconds.")

    # 7. Verification Report
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("TRANSACTION FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print(f"Output Path:      {output_path}")
    print(f"Output Shape:     {feature_df.shape}")
    print(f"File Size:        {file_size_mb:.2f} MB")
    print(f"Execution Time:   {elapsed:.2f} seconds")
    print("-" * 70)
    print("Features Inventory:")
    print(f"  * Metadata Key:   TransactionID")
    if target_cols:
        print(f"  * Target Label:   isFraud")
    print(f"  * Numerical:      {len(numerical_features)} columns (Amt, DT, dist, C, D, V)")
    print(f"  * Categorical:    {len(categorical_features)} columns (Product, card, addr, email, identity)")
    print(f"  * Derived:        3 columns (log_amount, card_combo, email_match)")
    print(f"  * Missingness:    5 columns (is_card_missing, etc.)")
    print("=" * 70)

if __name__ == "__main__":
    main()
