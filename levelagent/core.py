"""Level-generation agent for a letter-wheel word game (the WordTrip / Wordscapes format):
the player gets 5-7 letters and must find the hidden words that can be spelled from them.

    generate candidate  ->  critic (hard rules + difficulty)  ->  repair or reject  ->  log  ->  pack with a curve

Data: a frequency-ranked list of 20,000 English words (how common each word is), the ENABLE word-game dictionary
(172K valid words, no names or abbreviations), and 106 offensive words to keep out.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


class Lexicon:
    def __init__(self):
        freq = [w.strip() for w in (DATA / "20k.txt").read_text().split() if w.strip().isalpha()]
        valid = set((DATA / "enable1.txt").read_text().split())   # ENABLE: no names, no abbreviations
        self.banned = set((DATA / "banned.txt").read_text().split())
        self.rank = {w: i + 1 for i, w in enumerate(freq) if w in valid and 3 <= len(w) <= 7}
        self.words = list(self.rank)
        self.counts = {w: Counter(w) for w in self.words}
        self.banned_counts = {w: Counter(w) for w in self.banned if 3 <= len(w) <= 7}

    def formable(self, letters: str, pool=None):
        have = Counter(letters)
        pool = self.counts if pool is None else pool
        return [w for w, c in pool.items() if len(w) <= len(letters) and all(have[ch] >= n for ch, n in c.items())]


@dataclass
class Level:
    letters: str
    targets: list
    bonus: list = field(default_factory=list)       # valid but obscure words: accepted, never required
    difficulty: float = 0.0
    notes: list = field(default_factory=list)


@dataclass
class Rules:
    min_targets: int = 4
    max_targets: int = 12
    max_rank: int = 8000           # a required word rarer than this frustrates most players
    require_full_word: bool = True  # at least one target uses every letter (the satisfying "big word")


def difficulty(lex: Lexicon, letters, targets):
    """0-100. A level is as hard as its HARDEST required word (you must find them all), plus a little for how
    many words there are. The first version averaged word rarity instead; the player-model eval showed that
    score did not predict who finishes a level (correlation -0.09), so it was replaced with this one."""
    ranks = np.array([lex.rank[w] for w in targets])
    hardest = np.log10(ranks.max()) / np.log10(8000)            # 0..1
    second = np.log10(np.sort(ranks)[-2]) / np.log10(8000) if len(ranks) > 1 else hardest
    count = min(len(targets), 12) / 12
    return float(100 * (0.55 * hardest + 0.25 * second + 0.20 * count))


def critic(lex: Lexicon, lv: Level, rules: Rules, seen: set):
    """Returns a list of problems. Empty list = shippable."""
    p = []
    have = Counter(lv.letters)
    if any(any(have[c] < n for c, n in Counter(w).items()) for w in lv.targets):
        p.append("unsolvable: a target cannot be spelled from the letters")
    if lex.formable(lv.letters, lex.banned_counts):
        p.append("offensive word can be spelled from these letters")
    if any(lex.rank.get(w, 10**9) > rules.max_rank for w in lv.targets):
        p.append("obscure required word")
    if not rules.min_targets <= len(lv.targets) <= rules.max_targets:
        p.append("wrong number of targets")
    if rules.require_full_word and not any(len(w) == len(lv.letters) for w in lv.targets):
        p.append("no word uses every letter")
    if "".join(sorted(lv.letters)) in seen:
        p.append("duplicate of an earlier level")
    return p


class Agent:
    def __init__(self, lex=None, rules=None, seed=0):
        self.lex, self.rules = lex or Lexicon(), rules or Rules()
        self.rng = np.random.default_rng(seed)
        self.log = Counter()
        self.bases = {n: [w for w in self.lex.words if len(w) == n and self.lex.rank[w] <= 6000] for n in (5, 6, 7)}

    def naive(self, n_letters):
        """Baseline: random base word, every dictionary-valid formable word is required. No checks."""
        base = str(self.rng.choice(self.bases[n_letters]))
        letters = "".join(self.rng.permutation(list(base)))
        return Level(letters, sorted(self.lex.formable(letters), key=lambda w: (len(w), w)))

    def propose(self, n_letters, seen):
        """One candidate, repaired where possible, or None with the reason logged."""
        lv = self.naive(n_letters)
        r = self.rules
        common = [w for w in lv.targets if self.lex.rank[w] <= r.max_rank]
        if len(common) < len(lv.targets):                     # repair 1: obscure words become optional bonus words
            lv.bonus = [w for w in lv.targets if w not in common]; lv.targets = common
            lv.notes.append("moved obscure words to bonus")
        if len(lv.targets) > r.max_targets:                   # repair 2: keep the full-length word(s) + the commonest rest
            full = [w for w in lv.targets if len(w) == len(lv.letters)]
            rest = sorted((w for w in lv.targets if w not in full), key=lambda w: self.lex.rank[w])
            keep = full[:2] + rest[: r.max_targets - len(full[:2])]
            lv.bonus += [w for w in lv.targets if w not in keep]; lv.targets = sorted(keep, key=lambda w: (len(w), w))
            lv.notes.append("trimmed to the commonest targets")
        problems = critic(self.lex, lv, r, seen)
        if problems:
            for q in problems:
                self.log["rejected: " + q] += 1
            return None
        lv.difficulty = difficulty(self.lex, lv.letters, lv.targets)
        self.log["accepted"] += 1
        return lv

    def pack(self, n_levels, curve=None):
        """A pack whose difficulty follows a planned curve: rises overall, with an easier 'breather' every 5th
        level. Builds a pool of accepted levels, then assigns the closest unused level to each slot."""
        curve = curve if curve is not None else planned_curve(n_levels)
        seen, pool = set(), []
        while len(pool) < n_levels * 6:
            n = int(self.rng.choice([5, 6, 7], p=[0.3, 0.4, 0.3]))
            lv = self.propose(n, seen)
            if lv:
                seen.add("".join(sorted(lv.letters))); pool.append(lv)
        d = np.array([lv.difficulty for lv in pool])
        lo, hi = np.percentile(d, [3, 97])
        want = lo + (hi - lo) * curve
        used, out = set(), []
        for w in want:
            order = np.argsort(np.abs(d - w))
            i = next(int(j) for j in order if int(j) not in used)
            used.add(i); out.append(pool[i])
        return out, want


def planned_curve(n):
    base = np.linspace(0.05, 0.95, n)
    dip = np.array([0.22 if (i + 1) % 5 == 0 else 0.0 for i in range(n)])
    return np.clip(base - dip, 0, 1)


def player_completes(lex, lv, vocab_rank, rng, slip=0.03):
    """Independent check, not used by the generator: a player who knows the `vocab_rank` commonest words
    (with a small chance of missing any word) finishes a level only by finding every target."""
    return all(lex.rank[w] <= vocab_rank and rng.random() > slip for w in lv.targets)
