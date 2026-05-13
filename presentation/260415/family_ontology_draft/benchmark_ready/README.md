# Benchmark-Ready Family Package

This folder compresses the 29-family draft ontology into a benchmark-ready family shortlist
for complete control working-point evaluation.

## Selection Rule
- Keep every high-priority family with core_method_count >= 5. This yields 15 benchmark-ready topical families.
- Selected families: 15
- Excluded families: 14
- Budgets supported by the blueprint: 2%, 5%, 10%

## Files
- `benchmark_ready_family_definitions.yaml`: manual shortlist and per-family construction blueprint.
- `build_benchmark_ready_package.py`: regenerates the summary tables and markdown blueprint.
- `benchmark_ready_family_summary.csv`: selected family table with support counts and negative-family assignments.
- `excluded_family_notes.csv`: compressed-away families and why they are excluded from the first benchmark-ready pass.
- `target_control_blueprint.md`: detailed protocol and per-family target/control construction guidance.
- `family_retrieval_queries.yaml`: initial lexical retrieval query packs for all 15 benchmark-ready families.
- `family_annotation_guidelines.md`: shared labeling rubric for family-level target/control annotation.
- `family_annotation_sheet_template.csv`: header template for exported annotation batches.
- `build_family_level_dataset_skeleton.py`: candidate-mining and annotation-batch scaffold built on top of the blueprint.

## Global Benchmark Protocol
- Selection split: 40 target / 160 control.
- Calibration split: 500 control only.
- Evaluation split: 120 target / 500 control.
- Control mix:
  - hard negatives: 50%
  - medium negatives: 30%
  - background negatives: 20%

## Selected Families
| family_id | display_name | category | core_methods | exp_count | overlap_hits | overlap_risk |
| --- | --- | --- | ---: | ---: | ---: | --- |
| mlb_baseball | MLB / Professional Baseball | sports_and_games | 7 | 22 | 1 | low |
| nfl_football | NFL / American Football | sports_and_games | 6 | 21 | 8 | medium |
| soccer | Soccer / Association Football | sports_and_games | 7 | 20 | 1 | low |
| nba_basketball | NBA / Basketball | sports_and_games | 7 | 22 | 8 | medium |
| nhl_hockey | NHL / Hockey | sports_and_games | 6 | 21 | 0 | low |
| combat_sports | Combat Sports | sports_and_games | 6 | 17 | 0 | low |
| gaming_general | Video Games / Gaming | sports_and_games | 6 | 14 | 5 | medium |
| crypto_blockchain | Crypto / Blockchain | technology_and_science | 6 | 21 | 0 | low |
| aviation_aerospace | Aviation / Aerospace | technology_and_science | 5 | 18 | 0 | low |
| china | China | geography_and_politics | 6 | 21 | 0 | low |
| japan | Japan | geography_and_politics | 5 | 20 | 0 | low |
| russia_post_soviet | Russia / Post-Soviet Sphere | geography_and_politics | 7 | 22 | 0 | low |
| middle_east_geopolitics | Middle East Geopolitics | geography_and_politics | 6 | 22 | 2 | medium |
| us_electoral_politics | U.S. Electoral Politics | geography_and_politics | 5 | 19 | 1 | low |
| us_legislative_governance | U.S. Legislative / Governance | geography_and_politics | 5 | 14 | 0 | low |

## Excluded Families
| family_id | display_name | core_methods | overlap_hits | reason |
| --- | --- | ---: | ---: | --- |
| higher_education_campus | Higher Education / Campus | 4 | 12 | Only four core methods cover it, and the overlap queue is dominated by college-sports ambiguity. |
| judaism_jewish | Judaism / Jewish Life | 4 | 2 | Support is moderate, but the boundary with Middle East geopolitics is too entangled for the first benchmark-ready pass. |
| spirituality_new_age | Spirituality / Metaphysics | 4 | 10 | Only four core methods cover it, and overlap with Christianity is the largest unresolved collision in the draft. |
| entertainment_screen_media | Film / Television Media | 3 | 1 | Support is limited to three core methods, and the scope is broad enough to invite topical drift. |
| law_enforcement_crime | Law Enforcement / Crime | 3 | 1 | Coverage is too narrow and the semantics are mixed with institutional and court-reporting language. |
| interrogative_questions | Interrogative Questions | 3 | 0 | This is a discourse/form family, not a stable topical family for a control working-point benchmark. |
| reference_fragments | Referential Fragments | 4 | 0 | This family captures textual form rather than a semantic topic, so it is not suitable for the first family-level benchmark. |
| quoted_dialogue | Quoted Dialogue / Speech | 3 | 0 | This is a style family, not a topical family, and would not support meaningful hard-negative design. |
| women_feminine_subjects | Women / Feminine Subjects | 2 | 0 | The scope is broad and heterogeneous, and only two core methods cover it. |
| christianity | Christianity / Church Life | 2 | 10 | Coverage is low and ten overlap cases with spirituality indicate that the family needs more ontology cleanup first. |
| lgbtq | LGBTQ Topics | 2 | 0 | Coverage is too narrow for the first benchmark-ready set. |
| animals_pets | Animals / Pets | 2 | 0 | The family is under-supported and does not yet justify a dedicated benchmark slot. |
| tabletop_gaming | Tabletop / Card Gaming | 2 | 4 | This family is nested under gaming_general and is too under-covered to keep as a standalone benchmark family. |
| function_word_patterns | Function-Word / Grammatical Patterns | 1 | 0 | This is a grammatical-pattern family with one-method support, not a semantic benchmark topic. |

Regenerate with:

```bash
uv run python build_benchmark_ready_package.py
```
