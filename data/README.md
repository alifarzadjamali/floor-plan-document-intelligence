# Data

CubiCasa5K is not committed to this repository. It is licensed separately under
CC BY-NC 4.0 and is downloaded from Zenodo record 2613548.

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

The downloader flattens the archive's redundant top-level directory. The
expected local layout is `data/cubicasa5k/{train,val,test}.txt` plus the
sample directories referenced by those official split files. Each sample must
contain `F1_scaled.png` and `model.svg`.

The split files are copied verbatim from the archive. Development and parameter
selection use only train/validation. The test split is evaluated only after the
baseline configuration is frozen.
