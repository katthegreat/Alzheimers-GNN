from pathlib import Path
import pandas as pd

data_dir = Path("data/raw/oasis3/scans")
files = list(data_dir.glob("MRI-json*/resources/csv/files/*.csv"))

if not files:
    raise FileNotFoundError("MRI metadata CSV not found.")

df = pd.read_csv(files[0], low_memory=False)
df.columns = df.columns.str.strip()

print("Total MRI records:", len(df))
print("Unique participants:", df["subject_id"].nunique())

# Columns - These are the columsn that actually exist 
possible_columns = [
    "scan_category",
    "SeriesDescription",
    "SequenceName",
    "ScanningSequence",
    "ImageType",
    "label"
]

columns = [c for c in possible_columns if c in df.columns]

print("\nAvailable scan-description columns:", columns)

for col in columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False).head(20).to_string())

#  Keywords to find  the potential functional MRI records
keywords = r"rest|fmri|bold|functional|rsfmri|rs-fmri"

if not columns:
    raise ValueError("No scan-description columns found.")

mask = (
    df[columns]
    .fillna("")
    .astype(str)
    .apply(
        lambda col: col.str.contains(
            keywords, case=False, regex=True
        )
    )
    .any(axis=1)
)

fmri_scans = df.loc[mask].copy()

print("\nPotential fMRI records:", len(fmri_scans))
print("Participants:", fmri_scans["subject_id"].nunique())

print("\nMatching scan descriptions:")
print(
    fmri_scans[columns]
    .drop_duplicates()
    .head(20)
    .to_string(index=False)
)

output = Path("data/processed/potential_fmri_scans.csv")
output.parent.mkdir(parents=True, exist_ok=True)
fmri_scans.to_csv(output, index=False)

print("\nSaved to:", output)
