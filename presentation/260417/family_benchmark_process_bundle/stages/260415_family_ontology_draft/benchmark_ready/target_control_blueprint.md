# Family-Level Target / Control Blueprint

This document translates the benchmark-ready shortlist into a concrete data-construction
plan for a family-level complete control working-point benchmark.

## Global Protocol

### Final corpus policy
- All final target and control sentences must come from the same sentence-level corpus used for activation capture.
- Top activating sentences, parsed_responses summaries, and external web knowledge may be used only for seed discovery, alias expansion, and quality review.
- Do not place seed sentences taken from matched features directly into the final evaluation split.

### Split recipe
- Selection split: 40 target + 160 control.
- Calibration split: 500 control only.
- Evaluation split: 120 target + 500 control.

### Control mix
- Hard negatives: 50%
- Medium negatives: 30%
- Background negatives: 20%

### Construction guardrails
- Each method may submit at most one feature per family; choose it using only the selection split.
- Calibrate thresholds on the calibration-control split only, separately for each budget alpha.
- Report target reject rate and held-out control reject rate on the evaluation split.
- Use document-level or source-level deduplication whenever source metadata exists; otherwise apply normalized-text and high n-gram-overlap deduplication.
- When feasible, ensure at least 25% of evaluation-target sentences do not contain the canonical family label verbatim and instead rely on entities, jargon, or context.
- At least half of each control pool should be hard negatives from nearby semantic families rather than generic background text.

### Retrieval pipeline
- Stage 1: build family-specific seed lexicons from ontology aliases, matched feature summaries, and obvious entity names.
- Stage 2: retrieve a large candidate pool from the base corpus with lexical or hybrid retrieval.
- Stage 3: manually or LLM-assist label candidates against the family rubric.
- Stage 4: stratify negatives into hard, medium, and background buckets.
- Stage 5: split into selection, calibration, and evaluation partitions after deduplication.

## MLB / Professional Baseball (`mlb_baseball`)

- Category: sports_and_games
- Support: 7 core methods, 22 experiment outputs, 22 matched features
- Overlap review hits: 1
- Why kept: Maximal cross-method support and a tight league-defined scope make this a clean sports family for deployment-style benchmarking.

### Target definition
- Scope: Professional baseball with MLB teams, players, trades, injuries, game play, farm systems, and league administration.
- Seed terms:
  - mlb
  - major league baseball
  - baseball
  - world series
  - home run
  - pitcher
  - yankees
  - dodgers
- Exclusions:
  - college baseball
  - softball
  - cricket
  - generic multi-sport roundups where baseball is not primary

### Control construction
- Hard-negative benchmark families:
  - nfl_football
  - nba_basketball
  - nhl_hockey
  - soccer
- Hard negatives outside the benchmark shortlist:
  - college baseball
  - softball
  - cricket
- Medium-negative benchmark families:
  - combat_sports
  - gaming_general
- Background benchmark families:
  - crypto_blockchain
  - china
  - aviation_aerospace

### Construction notes
- Keep at least half of negatives inside sports journalism so the benchmark cannot be solved by a generic sports-writing detector.
- Do not admit pieces where baseball is only one bullet in a broad scoreboard roundup.

## NFL / American Football (`nfl_football`)

- Category: sports_and_games
- Support: 6 core methods, 21 experiment outputs, 28 matched features
- Overlap review hits: 8
- Why kept: Support is broad across methods, and the main ambiguity with college football is manageable with explicit NCAA hard negatives.

### Target definition
- Scope: Professional American football centered on the NFL, including games, teams, draft, free agency, coaching, injuries, and analytics.
- Seed terms:
  - nfl
  - super bowl
  - quarterback
  - touchdown
  - linebacker
  - draft
  - patriots
  - chiefs
- Exclusions:
  - college football
  - ncaa football
  - rugby
  - generic football references that resolve to soccer

### Control construction
- Hard-negative benchmark families:
  - soccer
  - mlb_baseball
  - nba_basketball
  - nhl_hockey
- Hard negatives outside the benchmark shortlist:
  - higher_education_campus
  - rugby
  - college football
- Medium-negative benchmark families:
  - combat_sports
  - gaming_general
- Background benchmark families:
  - crypto_blockchain
  - china
  - aviation_aerospace

### Construction notes
- Reserve a dedicated NCAA/college-football negative bucket because this is the dominant ambiguity observed in the draft ontology.
- Prefer target sentences about pro teams or league mechanics rather than generic mentions of football as a sport.

## Soccer / Association Football (`soccer`)

- Category: sports_and_games
- Support: 7 core methods, 20 experiment outputs, 25 matched features
- Overlap review hits: 1
- Why kept: This family is supported by every core method and is semantically stable across leagues, transfer news, and match analysis.

### Target definition
- Scope: Professional association football including club and international soccer, transfers, match reports, injuries, and tactical analysis.
- Seed terms:
  - soccer
  - premier league
  - champions league
  - transfer window
  - striker
  - manager
  - arsenal
  - fifa
- Exclusions:
  - american football
  - rugby
  - college sports
  - generic football mentions without soccer context

### Control construction
- Hard-negative benchmark families:
  - nfl_football
  - mlb_baseball
  - nba_basketball
  - nhl_hockey
- Hard negatives outside the benchmark shortlist:
  - rugby
  - higher_education_campus
  - gaelic football
- Medium-negative benchmark families:
  - combat_sports
  - gaming_general
- Background benchmark families:
  - china
  - crypto_blockchain
  - aviation_aerospace

### Construction notes
- Favor sentence pools from match reports, transfer journalism, and tactical analysis to preserve the style seen in matched features.
- Ensure hard negatives include other sports articles with player names and transfer-like business language.

## NBA / Basketball (`nba_basketball`)

- Category: sports_and_games
- Support: 7 core methods, 22 experiment outputs, 29 matched features
- Overlap review hits: 8
- Why kept: Full 7/7 method coverage and high recurrence make NBA content benchmark-worthy, with college-basketball confusion manageable via explicit negatives.

### Target definition
- Scope: Professional basketball centered on the NBA, including players, teams, trades, draft, playoffs, coaching, and statistical analysis.
- Seed terms:
  - nba
  - playoffs
  - trade deadline
  - point guard
  - lakers
  - celtics
  - draft
  - triple-double
- Exclusions:
  - college basketball
  - wnba
  - fiba-only coverage where nba is not primary
  - generic multi-sport roundups

### Control construction
- Hard-negative benchmark families:
  - nfl_football
  - mlb_baseball
  - nhl_hockey
  - soccer
- Hard negatives outside the benchmark shortlist:
  - higher_education_campus
  - college basketball
  - wnba
- Medium-negative benchmark families:
  - combat_sports
  - gaming_general
- Background benchmark families:
  - crypto_blockchain
  - china
  - aviation_aerospace

### Construction notes
- Keep a dedicated college-basketball negative slice because several draft matches mixed campus and pro-basketball language.
- Prefer NBA-specific transaction, analytics, and playoff language over generic basketball tips or drills.

## NHL / Hockey (`nhl_hockey`)

- Category: sports_and_games
- Support: 6 core methods, 21 experiment outputs, 21 matched features
- Overlap review hits: 0
- Why kept: The family is concrete, well supported, and easy to contrast against other pro team sports and non-hockey winter sports.

### Target definition
- Scope: Professional ice hockey centered on the NHL, covering teams, players, injuries, trades, standings, and game analysis.
- Seed terms:
  - nhl
  - stanley cup
  - power play
  - goalie
  - puck
  - overtime
  - maple leafs
  - canucks
- Exclusions:
  - field hockey
  - olympic or amateur hockey when nhl context is absent
  - generic winter-sports coverage

### Control construction
- Hard-negative benchmark families:
  - nba_basketball
  - mlb_baseball
  - nfl_football
  - soccer
- Hard negatives outside the benchmark shortlist:
  - field hockey
  - winter olympics general coverage
- Medium-negative benchmark families:
  - combat_sports
  - gaming_general
- Background benchmark families:
  - crypto_blockchain
  - china
  - aviation_aerospace

### Construction notes
- Require hockey as the primary topic, not a passing mention inside a broader sports page.
- Use other team-sport articles as the dominant negative style match.

## Combat Sports (`combat_sports`)

- Category: sports_and_games
- Support: 6 core methods, 17 experiment outputs, 17 matched features
- Overlap review hits: 0
- Why kept: The feature summaries are coherent around MMA and boxing, and the family offers strong same-register sports negatives.

### Target definition
- Scope: MMA, boxing, and related professional combat-sports coverage, including fighters, promotions, title fights, training, and matchmaking.
- Seed terms:
  - ufc
  - mma
  - boxing
  - knockout
  - title fight
  - heavyweight
  - octagon
  - bellator
- Exclusions:
  - wwe or scripted pro wrestling
  - general fitness
  - martial-arts philosophy without competitive context

### Control construction
- Hard-negative benchmark families:
  - nfl_football
  - nba_basketball
  - mlb_baseball
  - soccer
- Hard negatives outside the benchmark shortlist:
  - wwe / sports entertainment
  - fitness or workout advice
- Medium-negative benchmark families:
  - nhl_hockey
  - gaming_general
- Background benchmark families:
  - crypto_blockchain
  - china
  - aviation_aerospace

### Construction notes
- Keep both MMA and boxing in scope so the family stays broad enough to match the ontology.
- Hard negatives should still read like competitive-sports journalism, not generic unrelated text.

## Video Games / Gaming (`gaming_general`)

- Category: sports_and_games
- Support: 6 core methods, 14 experiment outputs, 24 matched features
- Overlap review hits: 5
- Why kept: Coverage reaches six core methods, and the family captures a useful non-news topical regime, but retrieval must aggressively exclude tabletop and generic statistics noise.

### Target definition
- Scope: Digital video games, esports, game development, platform ecosystems, reviews, patches, and gameplay systems.
- Seed terms:
  - video game
  - esports
  - patch notes
  - nintendo
  - playstation
  - steam
  - game developer
  - multiplayer
- Exclusions:
  - tabletop games
  - collectible card games without digital context
  - gambling or casino content
  - generic statistics articles that only resemble game data

### Control construction
- Hard-negative benchmark families:
  - combat_sports
  - nba_basketball
  - soccer
- Hard negatives outside the benchmark shortlist:
  - tabletop_gaming
  - entertainment_screen_media
  - gambling / casino
- Medium-negative benchmark families:
  - crypto_blockchain
  - japan
- Background benchmark families:
  - china
  - aviation_aerospace
  - us_legislative_governance

### Construction notes
- Require an explicit game, studio, platform, tournament, or gameplay mechanic to suppress the noisy quantitative-data false positive seen in the draft.
- Dedicate part of the hard-negative pool to tabletop and card-game writing so the family does not collapse into generic gaming jargon.

## Crypto / Blockchain (`crypto_blockchain`)

- Category: technology_and_science
- Support: 6 core methods, 21 experiment outputs, 21 matched features
- Overlap review hits: 0
- Why kept: This family has broad support and a well-defined technical and financial lexicon that is suitable for hard-negative stress tests.

### Target definition
- Scope: Cryptocurrency, blockchain protocols, tokens, exchanges, wallets, mining, smart contracts, and decentralized finance.
- Seed terms:
  - bitcoin
  - ethereum
  - blockchain
  - token
  - wallet
  - defi
  - smart contract
  - exchange
- Exclusions:
  - traditional stock-market reporting without crypto context
  - cybersecurity incidents that do not involve blockchain assets
  - generic fintech coverage

### Control construction
- Hard-negative benchmark families:
  - us_legislative_governance
  - china
  - gaming_general
- Hard negatives outside the benchmark shortlist:
  - traditional finance / equities
  - cybersecurity
  - payments infrastructure
- Medium-negative benchmark families:
  - aviation_aerospace
  - us_electoral_politics
- Background benchmark families:
  - mlb_baseball
  - soccer
  - japan

### Construction notes
- Hard negatives should share finance or technology register and may mention regulation, markets, or exchanges without any blockchain referent.
- Do not use only Bitcoin-heavy targets; include protocol, exchange, and DeFi coverage.

## Aviation / Aerospace (`aviation_aerospace`)

- Category: technology_and_science
- Support: 5 core methods, 18 experiment outputs, 22 matched features
- Overlap review hits: 0
- Why kept: The family is coherent across civil aviation and spaceflight, and it admits strong transportation and defense hard negatives.

### Target definition
- Scope: Aircraft, airlines, pilots, aviation safety, airports, rockets, satellites, missions, and aerospace industry operations.
- Seed terms:
  - aircraft
  - airline
  - pilot
  - nasa
  - spacex
  - rocket
  - satellite
  - launch
- Exclusions:
  - automotive or rail transport
  - maritime shipping
  - generic travel itineraries without aviation substance

### Control construction
- Hard-negative benchmark families:
  - china
  - russia_post_soviet
  - us_legislative_governance
- Hard negatives outside the benchmark shortlist:
  - maritime_naval
  - rail transport
  - automotive
- Medium-negative benchmark families:
  - crypto_blockchain
  - japan
- Background benchmark families:
  - mlb_baseball
  - soccer
  - combat_sports

### Construction notes
- Include both civil aviation and spaceflight, but require an aircraft or aerospace system to be central rather than incidental.
- Use transport and defense hard negatives so the family is not separable by generic technology-news style.

## China (`china`)

- Category: geography_and_politics
- Support: 6 core methods, 21 experiment outputs, 21 matched features
- Overlap review hits: 0
- Why kept: China appears across six core methods with clear entity and policy cues, making it a strong country-level family.

### Target definition
- Scope: China-related politics, economy, society, institutions, culture, geography, and international relations with China as the primary subject.
- Seed terms:
  - china
  - chinese
  - beijing
  - xi jinping
  - shanghai
  - ccp
  - renminbi
  - mainland
- Exclusions:
  - incidental references to chinese suppliers or products
  - taiwan-only coverage unless china is central
  - broader asia coverage where china is not primary

### Control construction
- Hard-negative benchmark families:
  - japan
  - russia_post_soviet
  - middle_east_geopolitics
  - us_legislative_governance
- Hard negatives outside the benchmark shortlist:
  - korea
  - taiwan-only politics
  - southeast asia general coverage
- Medium-negative benchmark families:
  - crypto_blockchain
  - aviation_aerospace
- Background benchmark families:
  - mlb_baseball
  - nba_basketball
  - gaming_general

### Construction notes
- Require China or a Chinese entity to be the primary frame, not a secondary trade partner mentioned in passing.
- Mix domestic, cultural, and international-relation targets so the family is not reducible to a single narrow subtopic.

## Japan (`japan`)

- Category: geography_and_politics
- Support: 5 core methods, 20 experiment outputs, 20 matched features
- Overlap review hits: 0
- Why kept: Japan has broad coverage and cleaner boundaries than the lower-support culture families, making it suitable for a country-level benchmark slot.

### Target definition
- Scope: Japan-related politics, economy, society, culture, geography, institutions, and Japan-focused international relations.
- Seed terms:
  - japan
  - japanese
  - tokyo
  - osaka
  - yen
  - diet
  - prime minister
  - nippon
- Exclusions:
  - anime or gaming references where japan is only incidental
  - east asia lists where japan is not primary
  - brand mentions with no japan-focused context

### Control construction
- Hard-negative benchmark families:
  - china
  - russia_post_soviet
  - us_electoral_politics
- Hard negatives outside the benchmark shortlist:
  - korea
  - anime-only entertainment
  - consumer electronics product lists
- Medium-negative benchmark families:
  - gaming_general
  - aviation_aerospace
- Background benchmark families:
  - mlb_baseball
  - combat_sports
  - crypto_blockchain

### Construction notes
- Require Japan or Japanese actors to be the primary subject of the sentence.
- Retain some cultural and society coverage, but exclude sentences where a Japanese proper noun is only decorative.

## Russia / Post-Soviet Sphere (`russia_post_soviet`)

- Category: geography_and_politics
- Support: 7 core methods, 22 experiment outputs, 24 matched features
- Overlap review hits: 0
- Why kept: This family has full 7/7 support and a stable geopolitical core spanning Russia, Ukraine, and the wider post-Soviet region.

### Target definition
- Scope: Russia, Ukraine, and the wider post-Soviet sphere, including domestic politics, military affairs, diplomacy, sanctions, and regional identity.
- Seed terms:
  - russia
  - russian
  - ukraine
  - kremlin
  - putin
  - moscow
  - sanctions
  - post-soviet
- Exclusions:
  - generic europe coverage without a post-soviet actor
  - cold-war history with no contemporary or regional link
  - slavic culture references without geographic relevance

### Control construction
- Hard-negative benchmark families:
  - china
  - middle_east_geopolitics
  - us_legislative_governance
- Hard negatives outside the benchmark shortlist:
  - european union general coverage
  - balkans outside the post-soviet sphere
- Medium-negative benchmark families:
  - aviation_aerospace
  - crypto_blockchain
- Background benchmark families:
  - mlb_baseball
  - gaming_general
  - combat_sports

### Construction notes
- Allow Ukraine and other former Soviet states, but keep the family anchored to the post-soviet geopolitical sphere.
- Use other international-news families as hard negatives, not just unrelated background.

## Middle East Geopolitics (`middle_east_geopolitics`)

- Category: geography_and_politics
- Support: 6 core methods, 22 experiment outputs, 45 matched features
- Overlap review hits: 2
- Why kept: Support is broad and overlap is limited, but the family must be framed explicitly around regional state conflict and policy rather than religion or identity.

### Target definition
- Scope: State conflict, diplomacy, insurgency, and regional power politics across the Middle East, including Syria, Iran, Iraq, Gulf politics, and the Israeli-Palestinian conflict when geopolitics is central.
- Seed terms:
  - iran
  - syria
  - iraq
  - gaza
  - israeli-palestinian
  - saudi
  - assad
  - hamas
- Exclusions:
  - pure jewish religious or cultural life
  - christian or muslim theology
  - identity-only references to israel with no geopolitical frame

### Control construction
- Hard-negative benchmark families:
  - russia_post_soviet
  - china
  - us_electoral_politics
- Hard negatives outside the benchmark shortlist:
  - judaism_jewish
  - christianity
  - spirituality_new_age
- Medium-negative benchmark families:
  - aviation_aerospace
  - us_legislative_governance
- Background benchmark families:
  - mlb_baseball
  - gaming_general
  - crypto_blockchain

### Construction notes
- Admit Israel-related sentences only when conflict, diplomacy, settlements, military action, or state policy is primary.
- Use religion and identity families as explicit hard negatives to keep the scope geopolitical.

## U.S. Electoral Politics (`us_electoral_politics`)

- Category: geography_and_politics
- Support: 5 core methods, 19 experiment outputs, 25 matched features
- Overlap review hits: 1
- Why kept: The family is broad enough to recur across methods and sharply aligned with campaigns, polling, voting, and candidate competition.

### Target definition
- Scope: U.S. elections, campaigns, candidates, primaries, polling, campaign strategy, and voting mechanics.
- Seed terms:
  - campaign
  - candidate
  - primary election
  - polling
  - delegate
  - caucus
  - swing state
  - ballot
- Exclusions:
  - legislative bargaining after elections
  - generic governance with no electoral contest
  - crime or court reporting

### Control construction
- Hard-negative benchmark families:
  - us_legislative_governance
  - china
  - middle_east_geopolitics
- Hard negatives outside the benchmark shortlist:
  - law_enforcement_crime
  - court proceedings
  - state administration with no election content
- Medium-negative benchmark families:
  - russia_post_soviet
  - crypto_blockchain
- Background benchmark families:
  - mlb_baseball
  - soccer
  - gaming_general

### Construction notes
- Require electoral competition, polling, campaign strategy, or voting mechanics to be central.
- Keep governance-only Washington reporting in the negative pool to separate elections from institutions.

## U.S. Legislative / Governance (`us_legislative_governance`)

- Category: geography_and_politics
- Support: 5 core methods, 14 experiment outputs, 18 matched features
- Overlap review hits: 0
- Why kept: This family complements electoral politics with institution-focused language and is supported by five core methods.

### Target definition
- Scope: Congress, bills, committees, votes, executive appointments, agencies, regulatory processes, and federal governance mechanics.
- Seed terms:
  - congress
  - senate
  - house
  - committee
  - bill
  - filibuster
  - regulation
  - cabinet
- Exclusions:
  - campaigns, polls, and election-night coverage
  - local crime or court reporting
  - opinion pieces with no institutional process

### Control construction
- Hard-negative benchmark families:
  - us_electoral_politics
  - china
  - russia_post_soviet
- Hard negatives outside the benchmark shortlist:
  - law_enforcement_crime
  - state-level court administration
  - campaign-finance stories without institutional action
- Medium-negative benchmark families:
  - crypto_blockchain
  - aviation_aerospace
- Background benchmark families:
  - mlb_baseball
  - gaming_general
  - combat_sports

### Construction notes
- Require an institution, officeholder action, or formal policy process to be central.
- Use campaign journalism as a first-class hard negative rather than generic background.

## Usage

- Use the selection split to choose one feature per family and method.
- Use the calibration-control split to tune thresholds for each alpha budget.
- Use the evaluation split to report family-level target reject and held-out control reject.
- Aggregate across families only after every method has been evaluated on the same family set.
