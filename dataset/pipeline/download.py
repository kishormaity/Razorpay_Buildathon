import os
import shutil
import zipfile
import sys

def download_dataset():
    # Setup project directories
    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    raw_dir = os.path.join(current_dir, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        import kagglehub
    except ImportError:
        print("\n[ERROR] kagglehub library not installed! Run: pip install kagglehub")
        sys.exit(1)

    dataset_handle = "ayushcl/ieee-fraud-detection-zip"
    print(f"Downloading dataset '{dataset_handle}' via kagglehub...")
    try:
        # Download using kagglehub (handles authentication bypass for public datasets)
        download_path = kagglehub.dataset_download(dataset_handle)
        print(f"Dataset successfully downloaded to cache: {download_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Please check your internet connection and verify that the dataset is publicly accessible.")
        sys.exit(1)
        
    print(f"Copying files from cache to local project directory: {raw_dir}...")
    try:
        # Transfer files from the global cache into the local project folder
        for item in os.listdir(download_path):
            s = os.path.join(download_path, item)
            d = os.path.join(raw_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
            print(f" - Copied: {item}")
    except Exception as e:
        print(f"\n[ERROR] Failed to copy files: {e}")
        sys.exit(1)
        
    # Check if we copied any nested zip files (e.g. train_transaction.csv.zip) and extract them
    print("Checking for zipped data archives...")
    for item in os.listdir(raw_dir):
        if item.endswith('.zip'):
            zip_file_path = os.path.join(raw_dir, item)
            print(f"Found archive: {item}. Extracting...")
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_dir)
                print(f"Successfully extracted: {item}. Cleaning up archive file...")
                os.remove(zip_file_path)
            except Exception as e:
                print(f"Warning: Failed to extract {item}: {e}")
                
    print("\nDataset successfully acquired! Run: python dataset/preprocess.py next.")

if __name__ == "__main__":
    download_dataset()
