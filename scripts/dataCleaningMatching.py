import pandas as pd

# Load your files
uids_df = pd.read_csv("aumc_peaks200_uids.csv", header=None, names=["uid"])
labels_df = pd.read_csv("aumcOutcomeData.csv", sep=";", quotechar='"')

# Merge on UID
merged_df = pd.merge(
    uids_df,
    labels_df,
    left_on="uid",
    right_on="AnalyteUid",
    how="inner"  # keep only spectra that have labels
)

# Drop rows with missing ESBL
merged_df = merged_df.dropna(subset=["ESBL"])

# Map label: "S" → 0 (susceptible), others (e.g. "R") → 1 (resistant)
merged_df["label"] = (merged_df["ESBL"] != "S").astype(int)

# Save cleaned label file
merged_df[["uid", "label"]].to_csv("aumc_cleaned_labels_esbl.csv", index=False)
