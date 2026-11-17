import numpy as np

from levelagent.core import Agent, Level, Lexicon, Rules, critic, difficulty

LEX = Lexicon()


def test_critic_flags_each_problem():
    r = Rules()
    assert any("unsolvable" in p for p in critic(LEX, Level("abc", ["cab", "zebra", "abc", "bac"]), r, set()))
    assert any("duplicate" in p for p in critic(LEX, Level("stone", ["one", "ton", "net", "stone"]), r, {"enost"}))
    assert any("wrong number" in p for p in critic(LEX, Level("stone", ["stone"]), r, set()))
    assert any("no word uses" in p for p in critic(LEX, Level("stone", ["one", "ton", "net", "ten"]), r, set()))


def test_offensive_letters_are_rejected_even_if_not_a_target():
    bad = sorted(LEX.banned_counts, key=len)[0]
    lv = Level(bad + "e", ["xx"])
    assert any("offensive" in p for p in critic(LEX, lv, Rules(), set()))


def test_pack_is_clean_and_follows_the_curve():
    a = Agent(LEX, seed=3)
    pack, want = a.pack(40)
    seen = set()
    for lv in pack:
        assert critic(LEX, lv, Rules(), seen) == []
        seen.add("".join(sorted(lv.letters)))
    d = np.array([lv.difficulty for lv in pack])
    assert np.corrcoef(d, want)[0, 1] > 0.95
    assert all(pack[i].difficulty < pack[i - 1].difficulty for i in range(4, 40, 5))      # breather levels


def test_harder_word_means_harder_level():
    assert difficulty(LEX, "stone", ["one", "ton", "net", "stone"]) < difficulty(LEX, "globe", ["gel", "leg", "log", "globe"])
