from modules.persona_voice import LIMITS, voice_for

QUIET = {"O": 0, "C": 0.9, "E": -0.8, "A": 0.5, "N": 0}
LIVELY = {"O": 0, "C": 0, "E": 0.9, "A": 0, "N": 0.4}


def _n(v):
    return float(v.rstrip("%dBst"))


def test_extravert_is_faster_higher_brighter_than_introvert():
    q, l = voice_for("companion", QUIET), voice_for("companion", LIVELY)
    for key in ("rate", "pitch", "tone", "volume"):
        assert _n(l[key]) > _n(q[key]), key


def test_role_changes_the_voice_for_the_same_personality():
    assert voice_for("receptionist", LIVELY) != voice_for("storyteller", LIVELY)


def test_values_stay_within_the_limits_and_use_the_clients_units():
    v = voice_for("tutor", {k: 1.0 for k in "OCEAN"})
    w = voice_for("tutor", {k: -1.0 for k in "OCEAN"})
    for out in (v, w):
        assert out["rate"].endswith("%") and out["pitch"].endswith("st")
        assert out["tone"].endswith("dB") and out["volume"].endswith("dB")
        for key, lim in LIMITS.items():
            assert abs(_n(out[key])) <= lim + 0.05


def test_unknown_role_and_missing_traits_do_not_crash():
    assert voice_for("something else", {}) == {"rate": "+0%", "pitch": "+0.0st",
                                              "tone": "+0dB", "volume": "+0dB"}


def test_companion_storyteller_tutor_share_one_voice_other_roles_are_other_people():
    from modules.persona_voice import CHARACTER_VOICE, ROLE_VOICE
    calm = {k: 0.0 for k in "OCEAN"}
    base = [voice_for(r, calm) for r in ROLE_VOICE]
    assert all(abs(_n(v["pitch"])) <= 1 and "gender" not in v for v in base)     # same base voice, small nudges
    man, boy = voice_for("receptionist", calm), voice_for("reading buddy", calm)
    assert _n(man["pitch"]) <= -4 and _n(boy["pitch"]) >= 4 and man["gender"] == boy["gender"] == "male"
    assert set(CHARACTER_VOICE) == {"receptionist", "reading buddy"}
    for ocean in ({k: 1.0 for k in "OCEAN"}, {k: -1.0 for k in "OCEAN"}):       # still bounded for ffmpeg
        assert abs(_n(voice_for("reading buddy", ocean)["pitch"])) <= 7 + 0.05
