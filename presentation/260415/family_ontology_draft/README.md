# Family Ontology Draft

This folder contains an editable first-pass family ontology built from
`Thesis/exp/**/output/l22/output/parsed_responses.yaml`.

## Scope
- Source files scanned: 23
- Total L22 feature records: 1140
- Families in shortlist: 29
- Records with at least one family match: 553
- Overlap review queue size: 31

## Files
- `family_seed_definitions.yaml`: editable seed definitions and regex patterns.
- `build_family_ontology_draft.py`: regeneration script.
- `family_ontology_draft.yaml`: generated ontology payload with counts and examples.
- `family_summary.csv`: compact family-level summary table.
- `family_coverage_by_experiment.csv`: feature-count matrix across experiment outputs.
- `family_coverage_by_core_method.csv`: feature-count matrix across the 7 core methods.
- `family_match_details.csv`: one matched feature-family row per hit.
- `family_overlap_review.csv`: records matching multiple families and needing manual review.

## Current Shortlist
| family_id | category | priority | exp_count | core_methods | matched_features |
| --- | --- | --- | ---: | --- | ---: |
| mlb_baseball | sports_and_games | high | 22 | topk,batchtopk,relu,gatedsae,jumprelu,dense,kernel | 22 |
| nfl_football | sports_and_games | high | 21 | batchtopk,relu,gatedsae,jumprelu,dense,kernel | 28 |
| soccer | sports_and_games | high | 20 | topk,batchtopk,relu,gatedsae,jumprelu,dense,kernel | 25 |
| nba_basketball | sports_and_games | high | 22 | topk,batchtopk,relu,gatedsae,jumprelu,dense,kernel | 29 |
| nhl_hockey | sports_and_games | high | 21 | topk,batchtopk,relu,gatedsae,dense,kernel | 21 |
| combat_sports | sports_and_games | high | 17 | topk,batchtopk,relu,gatedsae,dense,kernel | 17 |
| gaming_general | sports_and_games | high | 14 | batchtopk,relu,gatedsae,jumprelu,dense,kernel | 24 |
| tabletop_gaming | sports_and_games | high | 6 | gatedsae,dense | 7 |
| crypto_blockchain | technology_and_science | high | 21 | topk,batchtopk,relu,gatedsae,dense,kernel | 21 |
| aviation_aerospace | technology_and_science | high | 18 | topk,batchtopk,relu,dense,kernel | 22 |
| china | geography_and_politics | high | 21 | topk,batchtopk,relu,gatedsae,dense,kernel | 21 |
| japan | geography_and_politics | high | 20 | batchtopk,relu,gatedsae,dense,kernel | 20 |
| russia_post_soviet | geography_and_politics | high | 22 | topk,batchtopk,relu,gatedsae,jumprelu,dense,kernel | 24 |
| middle_east_geopolitics | geography_and_politics | high | 22 | topk,batchtopk,relu,gatedsae,dense,kernel | 45 |
| judaism_jewish | geography_and_politics | high | 16 | batchtopk,relu,dense,kernel | 16 |
| us_electoral_politics | geography_and_politics | high | 19 | batchtopk,relu,gatedsae,jumprelu,kernel | 25 |
| us_legislative_governance | geography_and_politics | high | 14 | topk,relu,gatedsae,jumprelu,kernel | 18 |
| higher_education_campus | society_and_culture | high | 16 | batchtopk,relu,dense,kernel | 19 |
| law_enforcement_crime | society_and_culture | high | 12 | gatedsae,jumprelu,kernel | 16 |
| animals_pets | society_and_culture | high | 7 | dense,kernel | 7 |
| entertainment_screen_media | society_and_culture | high | 13 | batchtopk,dense,kernel | 18 |
| christianity | society_and_culture | high | 13 | dense,kernel | 18 |
| spirituality_new_age | society_and_culture | high | 13 | relu,gatedsae,dense,kernel | 13 |
| lgbtq | society_and_culture | high | 11 | dense,kernel | 11 |
| women_feminine_subjects | society_and_culture | high | 16 | topk,batchtopk | 16 |
| interrogative_questions | discourse_and_form | medium | 17 | relu,dense,kernel | 18 |
| quoted_dialogue | discourse_and_form | medium | 11 | topk,batchtopk,kernel | 14 |
| reference_fragments | discourse_and_form | medium | 15 | relu,gatedsae,jumprelu,kernel | 25 |
| function_word_patterns | discourse_and_form | medium | 9 | dense | 28 |

## Notes
- The shortlist intentionally keeps only medium-granularity families that recur across multiple outputs.
- Long-tail or artifact-heavy families such as `cybersecurity`, `maritime_naval`, `immigration_border`, and `cricket` are left out of this initial shortlist.
- Overlapping records are expected at this stage. Use `family_overlap_review.csv` to split or tighten definitions before building a family-level control benchmark.
