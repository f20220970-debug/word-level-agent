"""Generate packs with the agent and with a naive generator, then evaluate both.  python run.py (~1 min)"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from levelagent.core import Agent, Lexicon, Rules, critic, difficulty, planned_curve, player_completes

ROOT = Path(__file__).resolve().parent
lex = Lexicon()
N = 300
agent = Agent(lex, seed=0)
pack, want = agent.pack(N)
naive_gen = Agent(lex, seed=1)
naive = [naive_gen.naive(int(naive_gen.rng.choice([5, 6, 7], p=[0.3, 0.4, 0.3]))) for _ in range(N)]


def audit(levels, label):
    seen, bad = set(), {"offensive": 0, "obscure": 0, "count": 0, "no_full": 0, "dup": 0, "unsolvable": 0, "any": 0}
    for lv in levels:
        p = critic(lex, lv, Rules(), seen)
        seen.add("".join(sorted(lv.letters)))
        for k, key in (("offensive", "offensive"), ("obscure", "obscure"), ("wrong number", "count"), ("no word uses", "no_full"), ("duplicate", "dup"), ("unsolvable", "unsolvable")):
            bad[key] += any(k in q for q in p)
        bad["any"] += bool(p)
    return {"generator": label, **{f"pct_{k}": round(100 * v / len(levels), 1) for k, v in bad.items()},
            "avg_targets": round(float(np.mean([len(lv.targets) for lv in levels])), 1)}


rows = [audit(pack, "agent (generate, critic, repair)"), audit(naive, "naive (random word, all valid words)")]

# independent player model: 3 player types by vocabulary size, 200 simulated players each
rng = np.random.default_rng(7)
players = {"casual (knows 3K commonest words)": 3000, "regular (6K)": 6000, "strong (12K)": 12000}
comp = {}
for name, levels in (("agent", pack), ("naive", naive)):
    for pl, v in players.items():
        c = np.array([[player_completes(lex, lv, v, rng) for _ in range(60)] for lv in levels]).mean(axis=1)
        comp[(name, pl)] = c
diff_agent = np.array([lv.difficulty for lv in pack])
reg = comp[("agent", "regular (6K)")]
thirds = np.array_split(np.arange(N), 3)
S = {"levels_per_pack": N, "candidates_tried": int(sum(agent.log.values()) if False else agent.log["accepted"] + sum(v for k, v in agent.log.items() if k.startswith("rejected"))),
     "accepted": int(agent.log["accepted"]), "rejection_log": {k: int(v) for k, v in agent.log.most_common()},
     "curve_spearman_planned_vs_actual": round(float(spearmanr(want, diff_agent).correlation), 3),
     "difficulty_vs_regular_player_completion_spearman": round(float(spearmanr(diff_agent, reg).correlation), 3),
     "regular_player_completion_by_third_agent": [round(100 * float(reg[t].mean()), 1) for t in thirds],
     "regular_player_completion_by_third_naive": [round(100 * float(comp[("naive", "regular (6K)")][t].mean()), 1) for t in thirds],
     "completion_overall": {f"{n} | {p}": round(100 * float(c.mean()), 1) for (n, p), c in comp.items()},
     "breather_levels_easier_pct": round(100 * float(np.mean([pack[i].difficulty < pack[i - 1].difficulty for i in range(4, N, 5)])), 1),
     "audit": rows}
(ROOT / "results" / "summary.json").write_text(json.dumps(S, indent=1))
(ROOT / "results" / "pack.json").write_text(json.dumps([{"level": i + 1, "letters": lv.letters.upper(), "targets": lv.targets, "bonus": lv.bonus[:15],
                                                          "difficulty": round(lv.difficulty, 1), "notes": lv.notes} for i, lv in enumerate(pack)], indent=0))
print(json.dumps(S, indent=1))
for i in (0, 60, 150, 299):
    print(i + 1, pack[i].letters.upper(), round(pack[i].difficulty, 1), pack[i].targets)
