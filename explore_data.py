"""Explore both real datasets to understand their structure."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, pandas as pd

print("=" * 60)
print("  ISCX-URL-2016 (All.csv)")
print("=" * 60)
df = pd.read_csv('All.csv', nrows=10000)
print(f"  Shape: {df.shape}")
print(f"  Columns ({len(df.columns)}):")
for i, c in enumerate(df.columns):
    print(f"    {i:2d}: {c} (dtype={df[c].dtype})")
print(f"\n  Label column 'URL_Type_obf_Type':")
print(f"    Values: {df['URL_Type_obf_Type'].value_counts().to_dict()}")
print(f"  Any NaN: {df.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("  KAGGLE PHISHING EMAIL")
print("=" * 60)
kaggle_dir = os.path.expanduser('~/.cache/kagglehub/datasets/naserabdullahalam/phishing-email-dataset/versions/1')
for fname in sorted(os.listdir(kaggle_dir)):
    if fname.endswith('.csv'):
        try:
            edf = pd.read_csv(os.path.join(kaggle_dir, fname), nrows=5, encoding='utf-8', on_bad_lines='skip')
            print(f"\n  {fname}: shape={edf.shape}")
            print(f"    Columns: {list(edf.columns)}")
        except Exception as e:
            print(f"\n  {fname}: ERROR - {e}")

# Check the main phishing_email.csv in detail
print("\n" + "=" * 60)
print("  phishing_email.csv (main file)")
print("=" * 60)
edf = pd.read_csv(os.path.join(kaggle_dir, 'phishing_email.csv'), nrows=1000, encoding='utf-8', on_bad_lines='skip')
print(f"  Shape: {edf.shape}")
print(f"  Columns: {list(edf.columns)}")
for c in edf.columns:
    print(f"    {c}: dtype={edf[c].dtype}, sample={edf[c].iloc[0]}")
if 'Email Type' in edf.columns:
    print(f"  Label 'Email Type': {edf['Email Type'].value_counts().to_dict()}")
elif 'label' in edf.columns:
    print(f"  Label 'label': {edf['label'].value_counts().to_dict()}")
