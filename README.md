# Level-Generation Agent for a Word Game

[![tests](https://github.com/f20220970-debug/word-level-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/word-level-agent/actions/workflows/tests.yml)

**[Live demo](https://f20220970-debug.github.io/word-level-agent/)**: play levels the agent generated.

## In 30 seconds
- An agent that generates letter-wheel word levels (the WordTrip / Wordscapes format), **checks its own work with a critic, repairs or rejects**, and arranges a pack on a planned difficulty curve.
- In 300-level packs, **0% of the agent's levels have a problem vs 99% for a naive generator**; 11% of naive levels let the player spell an offensive word.
- A separate simulated player can finish **51% of the agent's levels vs 1%** of the naive ones, and that same check caught a flaw in my first difficulty score.

Built for **Chaos Engine Labs' PS-II project "Agentic systems for game content"** (levels, puzzles and quality
review, with guardrails, observability and evals). Companion to my
[experiment-review agent](https://github.com/f20220970-debug/game-experiment-agent) for the LiveOps project.

> **Data (real word lists):** 20,000 English words ranked by how common they are, the ENABLE word-game
> dictionary (172K valid words, no names or abbreviations) and 106 offensive words to keep out.

## How it works (`levelagent/core.py`)
```
generate candidate ─► critic ─► repair or reject ─► log the reason ─► place on the difficulty curve
```
| Critic rule (guardrail) | Why |
| --- | --- |
| Every required word can be spelled from the letters | A level that can't be finished is a refund request |
| **No offensive word can be formed from the letters**, required or not | Players find them, screenshot them and post them |
| No obscure required words (outside the 8,000 commonest) | One word nobody knows blocks the whole level |
| 4 to 12 required words; one uses every letter | Too few is dull, too many is a chore; the long word is the payoff |
| No repeat of an earlier letter set | Repeats feel cheap |

Repairs before rejecting: obscure words become optional **bonus words**; over-long lists are trimmed to the
commonest words. Every rejection is counted by reason (observability):

| Outcome over 9,406 candidates | Count |
| --- | --- |
| Accepted | 1,800 |
| Rejected: duplicate letter set | 5,673 |
| Rejected: offensive word formable | 1,151 |
| Rejected: wrong number of words | 782 |

## Evals (`run.py`)
**1. Quality audit, 300 levels each**

| | Agent | Naive generator (random word, every valid word required) |
| --- | --- | --- |
| Levels with any problem | **0%** | 99.3% |
| Offensive word formable | **0%** | 11.3% |
| Obscure required word | **0%** | 97.7% |
| Duplicate letter sets | **0%** | 7.0% |
| Average required words | 8.2 | 21.5 |

**2. Can players finish the levels?** A separate player model (not used by the generator): a player knows
the N commonest words and must find every required word.

| Player | Agent's levels finished | Naive levels finished |
| --- | --- | --- |
| Casual (knows 3,000 commonest words) | 8.6% | 0.6% |
| Regular (6,000) | **50.6%** | 1.2% |
| Strong (12,000) | 78.6% | 5.7% |

**3. Does the pack follow the planned curve?** Planned vs actual difficulty: rank correlation 1.00, and
100% of the "breather" levels (every fifth) are easier than the level before. For the regular player,
completion falls across the pack as intended: 71.5% in the first third, 44.7% in the middle, 35.5% in the last.

## Two mistakes the evals caught
1. **A difficulty score that didn't measure difficulty.** My first score averaged how rare a level's words
   were. Against the player model its correlation with completion was **−0.09**: useless. The reason is
   simple in hindsight: you must find *every* word, so a level is as hard as its hardest word. The new score
   (hardest and second-hardest word, plus count) correlates **−0.57**.
2. **A dictionary full of non-words.** Reading sample levels showed "required" words like INT, SCI and
   ERIC (abbreviations and names from a web-scraped list). Switching to the ENABLE word-game dictionary
   removed them.

## Run it
```bash
pip install -r requirements.txt
python run.py     # two 300-level packs + all evals (~1 min) -> results/summary.json, results/pack.json
pytest -q         # 4 tests: critic rules, offensive-letter check, clean pack on the curve, difficulty ordering
```

## Limitations and next steps
- The casual player finishes only 8.6% of levels: the 8,000-word limit is too loose for them. A real game
  would set it from player data, and that's the first thing I'd tune with Chaos Engine's numbers.
- The player model is a vocabulary cut-off, not real behaviour. With live data, difficulty becomes a
  prediction of completion rate, trained and checked against actual play.
- Next: an LLM step for level themes and hints, with the same critic as its guardrail, and A/B testing
  generated packs with the experiment-review agent.

Built by Pavitra Seth (BITS Pilani) as preparation for Chaos Engine Labs' PS-II. Not affiliated with Chaos
Engine Labs.
