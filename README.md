# TASTE Score Samples

Draft GitHub Pages visualization for TASTE preference scoring.

The current page is built from `battles_test.csv`, a test battle manifest with
pairwise human preferences. TASTE itself is pairwise, so per-image scores in the
UI are aggregate model outputs: the mean probability that an image wins against
the other candidates for the same prompt.

Build the snapshot:

```bash
PYTHONPATH=/home/ubuntu/taste/taste-scorer/src python scripts/build_test_demo.py
```

The earlier train-split prototype builder is still available:

```bash
python scripts/build_demo.py
```
