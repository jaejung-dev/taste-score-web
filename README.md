# TASTE Score Samples

Draft GitHub Pages visualization for TASTE preference scoring.

The page is built from a small reproducible snapshot of the public
`purvanshi/TASTE` train split. The public dataset does not currently expose a
separate validation split, so this should be treated as a demo/prototype rather
than an official validation report.

Build the snapshot:

```bash
python scripts/build_demo.py
```

Build and run TASTE pairwise scoring:

```bash
PYTHONPATH=/home/ubuntu/taste/taste-scorer/src python scripts/build_demo.py --score
```
