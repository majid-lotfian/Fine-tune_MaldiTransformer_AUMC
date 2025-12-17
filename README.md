# Fine-tuning MaldiTransformer on AUMC MALDI-TOF Data for ESBL Prediction
**Overview**
This repository contains a downstream modeling pipeline for fine-tuning a pretrained MaldiTransformer model on AUMC MALDI-TOF mass spectrometry data for binary ESBL (AMR) prediction.
The repository focuses exclusively on model fine-tuning and evaluation, assuming that dataset construction and MALDI preprocessing are handled upstream by a dedicated and validated pipeline.

**Scope of This Repository**
What this repository does:
Loads a pretrained MaldiTransformer model.
Consumes a prepared dataset artifact (features + labels).
Fine-tunes the transformer for a binary downstream task (ESBL).
Evaluates performance using accuracy, AUROC, and AUPRC.
Saves fine-tuned model weights.




**Assumptions About Input Data**
This pipeline assumes the existence of a dataset artifact with the following properties:

Feature file
    NumPy array of shape: (N, 200, 2)
    Each sample consists of: 200 peaks
    Each peak has (mz, intensity)

Example:
    aumc_peaks200.npy

Label file
    CSV file aligned row-by-row with the feature file.
    Binary labels:
        0 → ESBL negative (susceptible)
        1 → ESBL positive (resistant)

Example:
aumc_cleaned_labels_esblV2.csv

Alignment guarantee
    Row i in the feature array corresponds to row i in the label file.
    No reordering or grouping is performed inside this repository.
    If multiple spectra per isolate exist, grouping and split strategy must be enforced upstream.

Repository Structure and Python Files

1.downloadModel.py
    Purpose: one-time setup utility
    Downloads the pretrained MaldiTransformer weights from Hugging Face.
    Caches the model locally for reuse.
    Independent of AUMC data.
    Usage:
        python 1.downloadModel.py

2.dataCleaningMatching.py
    Purpose: legacy label construction
    Merges spectrum UIDs with outcome metadata.
    Maps ESBL labels to binary targets.
    Status:
        Used in earlier experiments.
        Superseded by upstream dataset builders.
        Kept only for reproducibility of legacy results.
        This script should not be part of the current pipeline.

3.extract_top200_peaks.py
    Purpose: legacy feature extraction
    Extracts top-200 peaks from preprocessed .txt spectra.
    Produces .npy feature array and UID list.
    Status:
        Used before upstream preprocessing was validated.
        Must be replaced by upstream dataset construction.
        Kept for historical reference only.

4.1.train_maldi_amrV2.py
  Purpose: downstream fine-tuning and evaluation
  This is the only training script that should be used.
  Key characteristics:
      Loads a pretrained MaldiTransformer encoder.
      Uses CLS-token representation.
      Applies numerical-stability normalization.
      Supports cross-entropy or focal loss.
      Computes accuracy, AUROC, and AUPRC.
      Saves fine-tuned model weights.
  Usage:
      python 1.1.train_maldi_amrV2.py
  
  This script assumes a valid dataset artifact is already available.

**Conceptual Pipeline**

Step 1 — Dataset construction (UPSTREAM, external)
    Handled outside this repository:
    Raw MALDI parsing
    Preprocessing and validation
    Feature representation
    Label alignment
    Optional group-aware splitting

Output: dataset artifact compatible with this repository.

Step 2 — Model acquisition
    python downloadModel.py

Step 3 — Fine-tuning and evaluation
    python 1.1.train_maldi_amrV2.py

Outputs
    Printed evaluation metrics (loss, accuracy, AUROC, AUPRC).

Saved model:
    ../models/maldi_classifierV2.pth

Reproducibility Notes
    Random seeds are fixed.
    Results depend entirely on upstream dataset correctness.
    Any preprocessing changes require regenerating the dataset artifact.
    Deprecated scripts are retained only for historical comparison.
