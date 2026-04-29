# Family Annotation Guidelines

Use this template to manually label candidate sentences mined for the family-level control benchmark.

## Goal

Each sentence should be assigned to exactly one final label for the current family under review.
The final labels are used to build the family-specific `Target` pool and the three `Control` buckets.

## Final Labels

- `target_positive`: the sentence is clearly in-family and the family is the primary semantic frame.
- `hard_negative`: the sentence is clearly out-of-family but is a nearby confounder that should remain difficult.
- `medium_negative`: the sentence is out-of-family but still topically related enough to be non-trivial.
- `background_negative`: the sentence is cleanly out-of-family and mainly serves as background control.
- `ambiguous`: the sentence is too mixed, multi-topic, or underspecified to assign confidently.
- `drop`: the sentence is malformed, duplicate-like, too short, or otherwise unusable.

## Keep / Drop Rule

- Set `keep_for_benchmark=yes` only for `target_positive`, `hard_negative`, `medium_negative`, or `background_negative`.
- Set `keep_for_benchmark=no` for `ambiguous` and `drop`.

## Decision Rules

- The family must be the main topic, not a passing mention.
- Named entities alone are not enough if the surrounding sentence is really about another family.
- If a sentence mixes two benchmark families equally, mark `ambiguous`.
- If a sentence matches the family's explicit exclusion list, do not mark `target_positive`.
- Preserve difficult negatives. Do not over-clean the hard-negative bucket.

## Confidence Scale

- `high`: the sentence is straightforward and would likely be labeled the same by another annotator.
- `medium`: the label is probably right but depends on some contextual interpretation.
- `low`: the sentence is borderline and should be reviewed again before final benchmark inclusion.

## Suggested Workflow

1. Review all rows for one family at a time.
2. Start with rows proposed as `positive`.
3. Then review `benchmark_hard_negative` and `local_hard_negative`.
4. Finally review `medium_negative` and `background_negative`.
5. After labeling, remove exact duplicates and obvious near-duplicates before sampling splits.

## Required Columns

- `candidate_id`
- `family_id`
- `proposed_bucket`
- `source_row_index`
- `text`
- `matched_query_blocks`
- `matched_terms`
- `matched_family_ids`
- `final_label`
- `keep_for_benchmark`
- `label_confidence`
- `primary_reason`
- `exclusion_trigger`
- `notes`

## Primary Reason Suggestions

- `clear_in_family_anchor`
- `entity_plus_context`
- `nearby_confounder`
- `topical_but_not_primary`
- `too_ambiguous`
- `duplicate_or_low_quality`

