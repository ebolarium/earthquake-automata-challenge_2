# EarthArXiv Manuscript

This directory contains the English EarthArXiv manuscript source, generated
figures, submission metadata, and the deterministic PDF build script.

Build from the repository root with Python 3.11:

```bash
python -m pip install ".[paper]"
python paper/build_manuscript.py
```

The single-file submission artifact is written to:

```text
output/pdf/causal-spatial-correction-etas-preprint-v1.0.pdf
```

The PDF includes the EarthArXiv coversheet, manuscript, figures, declarations,
and references. Do not upload the figures as separate supplemental files.
