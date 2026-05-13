# Source Corpus Reference

The derived benchmark package in this folder includes all benchmark construction stages
and the final released benchmark splits, but it does **not** copy the original raw
base corpus parquet itself.

The source corpus used for candidate mining was:

- `/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/dataset/train-00000-of-00082.parquet`

Why it is not copied here:
- It is the unchanged upstream source corpus rather than a derived benchmark artifact.
- The benchmark-relevant outputs produced from it are already included in this bundle:
  - candidate pools
  - annotation batches
  - frozen benchmark pool
  - final released splits

If you need to rerun candidate mining from scratch, point:

- `stages/260415_family_ontology_draft/benchmark_ready/build_family_level_dataset_skeleton.py`

at the parquet above with `--dataset_format parquet --text_field text`.
