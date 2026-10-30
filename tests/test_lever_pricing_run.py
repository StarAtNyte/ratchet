"""lever_pricing_run: the plateau-diagnosis pricing instrument."""
import json

import pytest

import forgekernel.skills.lever_pricing_run as lpr
from forgekernel.loop.driver import parse_lever_prices

BASE = "import x\n\nA = 1\nB = 2\nC = 3\n"


def test_build_variants_composes_and_refuses_non_unique_or_noop_edits():
    levers = [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
              {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]},
              {"name": "c", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}]}]
    vs = lpr.build_variants(BASE, levers)
    assert list(vs) == ["base", "+a", "+b", "+c", "all", "all-a", "all-b", "all-c"]
    assert vs["all"] == "import x\n\nA = 10\nB = 20\nC = 30\n"
    assert vs["all-b"] == "import x\n\nA = 10\nB = 2\nC = 30\n"
    # an `old` that occurs twice is refused, named
    with pytest.raises(ValueError, match="lever 'dup' edit #1: `old` occurs 0 times"):
        lpr.build_variants(BASE, [{"name": "dup", "edits": [{"old": "Z = 9\n", "new": "Z = 1\n"}]}])
    with pytest.raises(ValueError, match="occurs 2 times"):
        lpr.build_variants(BASE + "A = 1\n", [{"name": "dup", "edits": [{"old": "A = 1\n", "new": "A = 2\n"}]}])
    # a lever whose edits cancel out is a no-op variant: refused, never priced as 0
    with pytest.raises(ValueError, match="byte-identical to base"):
        lpr.build_variants(BASE, [{"name": "noop", "edits": [{"old": "A = 1\n", "new": "A = 1\n"}]}])
    # composability: two levers editing the same text cannot both apply on `all`
    with pytest.raises(ValueError, match="lever 'a2' edit #1: `old` occurs 0 times"):
        lpr.build_variants(BASE, [{"name": "a1", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
                                  {"name": "a2", "edits": [{"old": "A = 1\n", "new": "A = 11\n"}]}])


def test_spread_and_median_helpers():
    assert lpr.med([3.0, 1.0, 2.0]) == 2.0 and lpr.med([None]) is None
    assert lpr.spread_pct([1.00, 1.02]) == pytest.approx(100 * 0.02 / 1.01)
    assert lpr.spread_pct([1.0]) == 0.0


def test_parse_lever_prices_reads_the_machine_lines():
    log = ("noise\nLEVER ring add=+9.23% remove=-9.22% add_ms=3.1314 remove_ms=3.228 "
           "class=independent spin_add=879610 spin_all_minus=991464 cubin_same_as_base=False\n"
           "LEVER nb add=+0.10% remove=n/a add_ms=3.40 remove_ms=None class=null "
           "spin_add=None spin_all_minus=None cubin_same_as_base=False\n"
           "STACK all=+16.16% sum_singles=+14.71% coupling=+1.46%\n"
           "PRICE-OK levers=2 rounds=3 base_ms=3.4056 all_ms=2.9866 spin_base=1126670 "
           "spin_all=800845 tensor_base=37.78 tensor_all=43.61\n")
    levers, stack, fail = parse_lever_prices(log)
    assert fail is None
    assert [l["name"] for l in levers] == ["ring", "nb"]
    assert levers[0]["add_pct"] == 9.23 and levers[0]["remove_pct"] == -9.22
    assert levers[0]["cls"] == "independent" and levers[0]["spin_add_ns"] == 879610
    assert levers[1]["remove_pct"] is None and levers[1]["remove_ms"] is None
    assert stack["all_pct"] == 16.16 and stack["coupling_pp"] == 1.46
    assert stack["base_ms"] == 3.4056 and stack["tensor_all"] == 43.61
    _, _, fail = parse_lever_prices("PRICE-FAIL reason=lever-edit (lever 'x' edit #1: `old` occurs 2 times)\n")
    assert fail.startswith("lever-edit")


def test_main_refuses_a_bad_spec_without_running_anything(tmp_path, monkeypatch, capsys):
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"levers": [{"name": "dup", "edits": [{"old": "nope\n", "new": "x\n"}]}]}))
    calls = []
    monkeypatch.setattr(lpr, "sh", lambda *a, **k: calls.append(a) or (0, "", ""))
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE-FAIL reason=lever-edit" in out and calls == []
    assert (root / "kernel.py").read_text() == BASE           # tree untouched
    (root / "notes" / "levers.json").write_text(json.dumps({"levers": []}))
    assert lpr.main(["notes/levers.json"]) == 0
    assert "PRICE-FAIL reason=no-levers" in capsys.readouterr().out


def test_a_group_of_edits_is_one_candidate_and_never_split():
    """The teacher groups edits that only work together into ONE lever; the
    tool prices the group as a unit (no per-edit variants), so a member that
    is worthless alone cannot be read as null."""
    grp = {"name": "grp", "edits": [{"old": "A = 1\n", "new": "A = 10\n"},
                                    {"old": "C = 3\n", "new": "C = 30\n"}]}
    solo = {"name": "solo", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]}
    vs = lpr.build_variants(BASE, [grp, solo])
    assert list(vs) == ["base", "+grp", "+solo", "all", "all-grp", "all-solo"]
    assert vs["+grp"] == "import x\n\nA = 10\nB = 2\nC = 30\n"       # both edits, together
    assert vs["all-grp"] == vs["+solo"]                                # removing the group removes both
    # edits inside a group apply in order, so a later edit may depend on an earlier one
    chain = {"name": "chain", "edits": [{"old": "A = 1\n", "new": "A = 1\nA2 = 5\n"},
                                        {"old": "A2 = 5\n", "new": "A2 = 50\n"}]}
    assert lpr.build_variants(BASE, [chain])["+chain"] == "import x\n\nA = 1\nA2 = 50\nB = 2\nC = 3\n"


def test_ratio_reading_prefers_the_bench_contract_key():
    """The bench contract emits `vs_ref_device`; the legacy `vs_cublas_device`
    spelling is a fallback, never the primary key — otherwise the same-job
    paired ratio is silently never read and every price falls back to ms."""
    assert lpr.ratio_reading({"vs_ref_device": 0.91, "vs_cublas_device": 0.5}) == 0.91
    assert lpr.ratio_reading({"vs_cublas_device": 0.5}) == 0.5
    assert lpr.ratio_reading({"device_median_ms": 1.0}) is None


def test_lever_name_is_one_token():
    assert lpr.lever_name({"name": " ring per stage "}) == "ring_per_stage"
    assert lpr.lever_name({}) == "unnamed"


def _fake_bench(scores):
    def run_bench(variant, keep_cubin):
        r = scores[variant]
        if r is None:
            return {"correctness": {"pass": False}}
        return {"correctness": {"pass": True},
                "performance": {"device_median_ms": round(1.0 / r, 4), "vs_ref_device": r}}
    return run_bench


def test_main_prices_the_named_kernel_file_and_prints_timing_before_roles(tmp_path, monkeypatch, capsys):
    """kernel_file comes from the spec (the driver writes ctx.kernel_file there);
    the LEVER / STACK table is printed as soon as timing is done (PRICE-TIMED),
    then again with the role columns (PRICE-OK); a missing cubin dump is
    `unknown`, never `noop`; the tree is left as found."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kern_bwd.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"kernel_file": "kern_bwd.py", "rounds": 2, "levers": [
            {"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
            {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]}]}))
    # a: worth +6% alone and costs 6% when removed from the stack (independent);
    # b: nothing alone, but the stack loses 4% without it (coupled)
    monkeypatch.setattr(lpr, "run_bench", _fake_bench(
        {"base": 0.60, "+a": 0.636, "+b": 0.60, "all": 0.66, "all-a": 0.60, "all-b": 0.6336}))
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    priced = []
    monkeypatch.setattr(lpr, "run_price", lambda v, **_: priced.append(v) or
                        {"tensor_active": 40.0, "roles": {"spin": {"exposed_ns": 100.0}}})
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE kernel_file=kern_bwd.py" in out
    assert out.index("PRICE-TIMED levers=2 rounds=2") < out.index("PRICE ROLES base") < out.index("PRICE-OK")
    assert out.count("STACK all=") == 2 and "cubin_same_as_base=unknown" in out
    levers, stack, fail = parse_lever_prices(out)
    assert fail is None and [l["name"] for l in levers] == ["a", "b"]
    assert levers[0]["cls"] == "independent" and levers[1]["cls"] == "coupled"
    assert levers[0]["spin_add_ns"] == 100.0          # the LAST copy (roles filled) wins
    assert stack["tensor_all"] == 40.0
    assert priced == ["base", "+a", "+b", "all", "all-a", "all-b"]
    assert (root / "kern_bwd.py").read_text() == BASE and not (root / "kernel.py").exists()


def test_main_stops_at_a_dead_base(tmp_path, monkeypatch, capsys):
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"levers": [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}]}))
    calls = []
    monkeypatch.setattr(lpr, "run_bench", lambda v, keep_cubin: calls.append(v) or None)
    monkeypatch.setattr(lpr, "run_price", lambda v, **_: (_ for _ in ()).throw(AssertionError("no roles")))
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE-FAIL reason=base-failed-correctness" in out and calls == ["base"]
    assert "LEVER " not in out and (root / "kernel.py").read_text() == BASE


def test_noop_needs_two_cubins_and_a_reading_inside_the_floor():
    """`noop` is a byte-identical +L cubin set; two missing dumps are not that.
    And identical cubins with a reading beyond the floor are not a noop either:
    a launch-config / host-side edit moves the time without touching a cubin."""
    levers = [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}]
    variants = lpr.build_variants(BASE, levers)
    order = list(variants)
    ok = {v: True for v in order}
    dev = {v: [1.0] for v in order}
    import io, contextlib
    for ratio_a, shas, expect in (
            (0.66, {"base": None, "+a": None}, "independent"),
            (0.601, {"base": "abc", "+a": "abc"}, "noop"),          # +0.17%: inside
            (0.66, {"base": "abc", "+a": "abc"}, "independent"),    # +10%: a real change
            (0.66, {"base": "abc", "+a": "def"}, "independent")):
        ratio = {"base": [0.60], "+a": [ratio_a], "all": [ratio_a]}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lpr._emit_table(levers, variants, order, ok, dev, ratio, shas, {}, final=True)
        assert f"class={expect}" in buf.getvalue(), (shas, buf.getvalue())


def test_cubin_sha_covers_every_dumped_cubin(tmp_path, monkeypatch):
    """A lever that changes a smaller compiled entry must not hash equal to base
    because the largest cubin did not move; file names / order do not matter."""
    def dump(variant, files):
        d = tmp_path / variant
        d.mkdir()
        for name, blob in files.items():
            (d / name).write_bytes(blob)
    monkeypatch.setattr(lpr, "_dump_dir", lambda v: str(tmp_path / v))
    big = b"K" * 4096
    dump("base", {"main.cubin": big, "aux.cubin": b"aux-v1"})
    dump("+a", {"main.cubin": big, "aux.cubin": b"aux-v2"})         # only the small one moved
    dump("+b", {"x.cubin": b"aux-v1", "y.cubin": big})               # same bytes, other names
    assert lpr.cubin_sha("+a") != lpr.cubin_sha("base")
    assert lpr.cubin_sha("+b") == lpr.cubin_sha("base")


def test_noise_floor_is_a_pooled_three_sigma_not_three_times_the_widest_range():
    """15 variants, 3 rounds, ~0.3% per-round jitter: the old rule (3 × the widest
    single range) put the floor near 3% and read a real +2% lever null; the pooled
    3σ floor on a difference of medians sits near 1%. One wild draw is named and
    kept out of the pool instead of setting the floor for the whole table."""
    import random
    rng = random.Random(7)
    order = ["base"] + [f"+l{i}" for i in range(14)]
    dev = {v: [1.0 * (1 + rng.gauss(0, 0.003)) for _ in range(3)] for v in order}
    ok = {v: True for v in order}
    old = max(3.0 * lpr.spread_pct(dev[v]) for v in order)
    new = lpr.noise_floor(order, ok, dev)
    assert new < 2.0 < old                                   # a +2% lever clears it now
    sigma, rounds, outliers = lpr.pooled_sigma(order, ok, dev)
    assert 0.15 < sigma < 0.5 and rounds == 3 and outliers == []
    dev["+l3"] = [1.0, 1.0, 6.5]                             # one wild draw
    sigma2, _, outliers2 = lpr.pooled_sigma(order, ok, dev)
    assert outliers2 == ["+l3"] and sigma2 == pytest.approx(sigma, rel=0.3)
    assert lpr.noise_floor(order, ok, dev) < 2.0


# --- the teacher chooses the combinations ------------------------------------------

def test_teacher_subsets_are_built_and_validated():
    """`variants` are the teacher's own combinations, built after the mechanical
    2N+2 block; a member must be a stack lever of the spec, names are one token
    and never a built variant's."""
    levers = [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
              {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]},
              {"name": "c", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}]}]
    vs = lpr.build_variants(BASE, levers, [{"name": "a b", "levers": ["a", "b"], "question": "does a need b?"}])
    assert list(vs)[-1] == "a_b" and vs["a_b"] == "import x\n\nA = 10\nB = 20\nC = 3\n"
    for bad, msg in [([{"name": "all", "levers": ["a"]}], "reserved"),
                     ([{"name": "a", "levers": ["a", "b"]}], "is a lever's name"),
                     ([{"name": "x", "levers": []}], "names no levers"),
                     ([{"name": "x", "levers": ["zz"]}], "unknown / solo"),
                     ([{"name": "x", "levers": ["a", "a"]}], "repeats"),
                     ([{"name": "x", "levers": ["a"]}, {"name": "x", "levers": ["b"]}], "duplicate subset")]:
        with pytest.raises(ValueError, match=msg):
            lpr.build_variants(BASE, levers, bad)
    # more than MAX_SUBSETS are dropped silently (the design turn was told the cap)
    many = [{"name": f"s{i}", "levers": ["a"]} for i in range(lpr.MAX_SUBSETS + 3)]
    assert sum(k.startswith("s") for k in lpr.build_variants(BASE, levers, many)) == lpr.MAX_SUBSETS


def test_a_solo_lever_is_priced_alone_and_kept_out_of_the_stack():
    """`in_stack: false` — the teacher suspects the lever harmful / exclusive:
    its +L is measured, it enters neither all nor all-L nor a subset."""
    levers = [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
              {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]},
              {"name": "c", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}], "in_stack": False}]
    vs = lpr.build_variants(BASE, levers)
    assert list(vs) == ["base", "+a", "+b", "+c", "all", "all-a", "all-b"]
    assert vs["all"] == "import x\n\nA = 10\nB = 20\nC = 3\n"
    with pytest.raises(ValueError, match="unknown / solo"):
        lpr.build_variants(BASE, levers, [{"name": "ac", "levers": ["a", "c"]}])
    # a solo lever's name is taken too: the driver's table is keyed by name
    with pytest.raises(ValueError, match="is a lever's name"):
        lpr.build_variants(BASE, levers, [{"name": "c", "levers": ["a", "b"]}])
    # every lever solo: single-adds only, no stack at all
    only = [dict(lv, in_stack=False) for lv in levers]
    assert list(lpr.build_variants(BASE, only)) == ["base", "+a", "+b", "+c"]


def test_main_times_subsets_and_buys_all_pos_when_a_lever_hurts_alone(tmp_path, monkeypatch, capsys):
    """A gdn table: one lever at -17.7% alone made all=-9% and every
    remove leg negative. The tool now (1) times the teacher's subsets, each as
    a VARIANT line, and (2) after the main pass, when a stack lever's single-add
    is below -noise and two stack levers remain, times `all-pos` = the stack
    without the harmful ones, reported on the STACK line."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"rounds": 2, "levers": [
            {"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
            {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]},
            {"name": "bad", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}]}],
         "variants": [{"name": "a+b", "levers": ["a", "b"], "question": "is a's gain b-dependent?"}]}))
    # a +6%, b +4%, bad -18%; all drags to -9%; a+b (= all-pos) +10%
    monkeypatch.setattr(lpr, "run_bench", _fake_bench(
        {"base": 0.60, "+a": 0.636, "+b": 0.624, "+bad": 0.492, "all": 0.546,
         "all-a": 0.51, "all-b": 0.52, "all-bad": 0.66, "a+b": 0.66, "all-pos": 0.66}))
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    monkeypatch.setattr(lpr, "run_price", lambda v, **_: {"tensor_active": 40.0, "roles": {"spin": {"exposed_ns": 5.0}}})
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE all-pos: stack minus bad (harmful alone) = a,b; second pass re-times base" in out
    assert "VARIANT a+b levers=a,b delta=+10.00%" in out and "q=is a's gain b-dependent?" in out
    assert "VARIANT all-pos levers=a,b delta=+10.00%" in out and "base re-timed at" in out
    assert "STACK all=-9.00%" in out and "all_pos=+10.00% all_pos_levers=a,b dropped=bad" in out
    # the second pass re-times base alongside all-pos, tagged as its own pass
    assert "PRICE ROUND 0 base pass=True" in out and "pass_tag=all-pos" in out
    assert out.count("base pass=True") == 4            # 2 rounds main + 2 rounds all-pos
    levers, stack, fail = parse_lever_prices(out)
    assert fail is None
    names = {l["name"]: l for l in levers}
    assert names["a+b"]["kind"] == "variant" and names["a+b"]["delta_pct"] == 10.0
    assert names["a+b"]["levers"] == ["a", "b"] and names["a+b"]["spin_ns"] == 5.0
    assert names["all-pos"]["kind"] == "variant"
    assert stack["all_pos_pct"] == 10.0 and stack["dropped"] == ["bad"] and stack["all_pos_levers"] == ["a", "b"]
    assert (root / "kernel.py").read_text() == BASE


def test_all_pos_is_not_bought_without_two_positive_levers_left(tmp_path, monkeypatch, capsys):
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"rounds": 1, "roles": False, "levers": [
            {"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
            {"name": "bad", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}]}]}))
    monkeypatch.setattr(lpr, "run_bench", _fake_bench(
        {"base": 0.60, "+a": 0.636, "+bad": 0.492, "all": 0.546, "all-a": 0.492, "all-bad": 0.636}))
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE all-pos" not in out and "VARIANT" not in out
    assert "all_pos=n/a all_pos_levers=none dropped=bad" in out


def test_harmful_and_unstacked_levers_are_not_null():
    """A lever whose single-add is a LOSS is `harmful` (the direction is wrong),
    not `null` (both ≈ 0); a positive single reading whose stack failed
    correctness is `unstacked`, not `independent` (|remove| was never measured)."""
    import io, contextlib
    levers = [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
              {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]}]
    variants = lpr.build_variants(BASE, levers)
    order = list(variants)
    ok = {v: True for v in order}
    dev = {v: [1.0] for v in order}
    ratio = {"base": [0.60], "+a": [0.55], "+b": [0.66], "all": [0.60],
             "all-a": [0.66], "all-b": [0.55]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lpr._emit_table(levers, variants, order, ok, dev, ratio, {}, {}, final=True)
    out = buf.getvalue()
    line_a = next(ln for ln in out.splitlines() if ln.startswith("LEVER a "))
    assert "add=-8.33% remove=+10.00%" in line_a and "class=harmful" in line_a
    assert "LEVER b add=+10.00%" in out and "class=independent" in out
    # the stack broke correctness: b's coupling is unmeasured
    ok2 = dict(ok, all=False)
    dev2 = dict(dev, all=[])
    ratio2 = dict(ratio, all=[])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lpr._emit_table(levers, variants, order, ok2, dev2, ratio2, {}, {}, final=True)
    line_b = next(ln for ln in buf.getvalue().splitlines() if ln.startswith("LEVER b "))
    assert "class=unstacked" in line_b
    line_a = next(ln for ln in buf.getvalue().splitlines() if ln.startswith("LEVER a "))
    assert "class=harmful" in line_a


def test_all_pos_delta_is_read_against_its_own_pass_base(tmp_path, monkeypatch, capsys):
    """Without a same-job reference the metric is device ms: node drift between
    the first pass and the all-pos pass must not land in the all-pos reading.
    The second pass re-times base and all-pos interleaved and reads all-pos
    against THAT base; the first pass's base stays the singles' base."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"rounds": 1, "roles": False, "levers": [
            {"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
            {"name": "b", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]},
            {"name": "bad", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}]}]}))
    # device ms only (no vs_ref_device); the node is 20% slower in the second pass
    ms = {"base": 1.00, "+a": 0.95, "+b": 0.96, "+bad": 1.20, "all": 1.10,
          "all-a": 1.15, "all-b": 1.14, "all-bad": 0.90}
    calls = []

    def run_bench(variant, keep_cubin):
        calls.append(variant)
        second_pass = calls.count("base") >= 2 if variant == "base" else variant == "all-pos"
        t = {"all-pos": 0.90}.get(variant, ms.get(variant, 1.0)) * (1.2 if second_pass else 1.0)
        return {"correctness": {"pass": True}, "performance": {"device_median_ms": round(t, 4)}}
    monkeypatch.setattr(lpr, "run_bench", run_bench)
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    # against the first pass's base (1.00 ms) the drifted all-pos (1.08 ms) would read -7.4%;
    # against its own pass's base (1.20 ms) it reads +11.11%
    assert "VARIANT all-pos levers=a,b delta=+11.11%" in out
    assert "all_pos=+11.11%" in out and "base re-timed at 1.2 ms" in out
    assert calls[-2:] == ["base", "all-pos"]
    assert "metric=device_ms" in out


def test_a_negative_single_add_the_stack_depends_on_is_coupled_not_harmful():
    """add < -noise alone does not make a lever harmful: when removing it from
    the stack costs (remove < -noise) the stack needs it — `coupled`. Only a
    lever that hurts alone AND is not missed by the stack is harmful, and only
    those are dropped for the all-pos pass."""
    import io, contextlib
    levers = [{"name": "ring", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]},
              {"name": "gran", "edits": [{"old": "B = 2\n", "new": "B = 20\n"}]},
              {"name": "bad", "edits": [{"old": "C = 3\n", "new": "C = 30\n"}]}]
    variants = lpr.build_variants(BASE, levers)
    order = list(variants)
    ok = {v: True for v in order}
    dev = {v: [1.0] for v in order}
    # ring +5% alone; gran -2% alone but the stack loses 6% without it; bad -3% alone
    # and the stack gains 4% without it
    ratio = {"base": [1.00], "+ring": [1.05], "+gran": [0.98], "+bad": [0.97],
             "all": [1.08], "all-ring": [0.98], "all-gran": [1.015], "all-bad": [1.123]}
    assert lpr.lever_class(-2.0, -6.25, noise=0.5) == "coupled"
    assert lpr.lever_class(-3.0, +4.0, noise=0.5) == "harmful"
    assert lpr.lever_class(-3.0, None, noise=0.5) == "harmful"
    assert lpr.lever_class(-3.0, -0.2, noise=0.5) == "harmful"      # removal ≈ 0: not missed
    auto = lpr.plan_all_pos(levers, variants, order, ok, dev, ratio)
    assert auto == {"levers": ["ring", "gran"], "dropped": ["bad"]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lpr._emit_table(levers, variants, order, ok, dev, ratio, {}, {}, final=True, auto=auto)
    lines = {ln.split()[1]: ln for ln in buf.getvalue().splitlines() if ln.startswith("LEVER ")}
    assert "class=coupled" in lines["gran"] and "add=-2.00%" in lines["gran"]
    assert "class=harmful" in lines["bad"] and "class=independent" in lines["ring"]
    assert "dropped=bad" in buf.getvalue()


# --- ablations: the conviction half of the same job -------------------------

_NCU_CSV = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size",'
    '"Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","1","python3","127.0.0.1","aux_kernel","1","7","(128, 1, 1)","(8, 1, 1)","0","9.0",'
    '"Command line profiler metrics","gpu__time_duration.sum","usecond","12.50"\n'
    '"1","1","python3","127.0.0.1","main_kernel","1","7","(384, 1, 1)","(132, 1, 1)","0","9.0",'
    '"Command line profiler metrics","gpu__time_duration.sum","msecond","3.10"\n'
    '"2","1","python3","127.0.0.1","main_kernel","1","7","(384, 1, 1)","(132, 1, 1)","0","9.0",'
    '"Command line profiler metrics","gpu__time_duration.sum","nsecond","3,000,000.00"\n'
    '"3","1","python3","127.0.0.1","aux_kernel","1","7","(128, 1, 1)","(8, 1, 1)","0","9.0",'
    '"Command line profiler metrics","gpu__time_duration.sum","usecond","12.40"\n')


def test_parse_ncu_time_csv_picks_the_dominant_kernel_across_units():
    """Dominant = the kernel NAME whose launches sum to the most time (a launch
    storm of small auxiliaries cannot out-vote the kernel); units normalise to
    ns and ncu's thousands separators are stripped."""
    ns, name, launches = lpr.parse_ncu_time_csv(_NCU_CSV)
    assert name == "main_kernel" and launches == 2
    assert ns == pytest.approx(3.05e6)
    assert lpr.parse_ncu_time_csv("no csv here\n") is None
    assert lpr.parse_ncu_time_csv(_NCU_CSV.splitlines()[0] + "\n") is None


def test_ablations_are_checked_and_applied_to_base_alone():
    levers = [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}]
    abl = [{"name": "no b", "hypothesis": "B costs", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]},
           {"name": "no_c", "edits": [{"old": "C = 3\n", "new": "C = 0\n"}]}]
    checked = lpr.check_ablations(abl, ["a"])
    assert [x["name"] for x in checked] == ["no_b", "no_c"]           # one token
    vs = lpr.build_ablation_variants(BASE, checked)
    assert list(vs) == ["~no_b", "~no_c"]
    assert vs["~no_b"] == "import x\n\nA = 1\nB = 0\nC = 3\n"          # base + this one only
    assert vs["~no_c"] == "import x\n\nA = 1\nB = 2\nC = 0\n"
    with pytest.raises(ValueError, match="shadows a lever or subset"):
        lpr.check_ablations([{"name": "a", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]}], ["a"])
    with pytest.raises(ValueError, match="reserved"):
        lpr.check_ablations([{"name": "all", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]}], [])
    with pytest.raises(ValueError, match="duplicate"):
        lpr.check_ablations([abl[1], abl[1]], [])
    with pytest.raises(ValueError, match="no edits"):
        lpr.check_ablations([{"name": "x"}], [])
    with pytest.raises(ValueError, match=f"max {lpr.MAX_ABLATIONS}"):
        lpr.check_ablations([dict(abl[1], name=f"n{i}") for i in range(lpr.MAX_ABLATIONS + 1)], [])
    with pytest.raises(ValueError, match="no-op"):
        lpr.build_ablation_variants(BASE, [{"name": "same", "edits": [{"old": "B = 2\n", "new": "B = 2\n"}]}])
    # a no-lever spec still builds (base only)
    assert list(lpr.build_variants(BASE, [])) == ["base"]


def _fake_ncu(times):
    def run_ncu_time(variant):
        t = times.get(variant)
        return None if t is None else (t, "main_kernel", 8)
    return run_ncu_time


def test_main_runs_the_ablation_pass_after_the_lever_table(tmp_path, monkeypatch, capsys):
    """Levers and ablations in one spec: the lever table is printed (both
    copies) BEFORE any ablation variant runs; each ablation is timed on the
    NCU clock against a base re-timed in the same pass, a broken one still
    reads (correctness=fail, device_delta=n/a), a hung one reads n/a; the
    kernel file is restored."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    spec = {"rounds": 1, "ablation_rounds": 2,
            "levers": [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}],
            "ablations": [
                {"name": "no_b", "hypothesis": "B is the exposed wait", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]},
                {"name": "hang", "edits": [{"old": "C = 3\n", "new": "C = 0\n"}]}]}
    (root / "notes" / "levers.json").write_text(json.dumps(spec))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    seen = []

    def run_bench(variant, keep_cubin, timeout=None):
        seen.append(("bench", variant, open(root / "kernel.py").read()))
        assert timeout == lpr.ABLATION_BENCH_TIMEOUT if variant.startswith("~") else True
        if variant == "~no_b":
            return {"correctness": {"pass": False}, "performance": None}
        return {"correctness": {"pass": True},
                "performance": {"device_median_ms": 3.0 if variant == "base" else 2.9,
                                "vs_ref_device": 0.6}}
    monkeypatch.setattr(lpr, "run_bench", run_bench)
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    monkeypatch.setattr(lpr, "run_price", lambda v, **_: seen.append(("price", v)) or None)
    monkeypatch.setattr(lpr, "run_ncu_time",
                        lambda v: seen.append(("ncu", v)) or _fake_ncu(
                            {"base": 1000.0, "~no_b": 900.0})(v))
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    # order: PRICE-OK (final lever table) before the first ABLATE ROUND
    assert out.index("PRICE-OK") < out.index("ABLATE ROUND 0")
    assert "LEVER a add=" in out
    # ncu rounds interleave base with every ablation, twice; ncu runs BEFORE the bench
    ncu = [v for k, v in [(x[0], x[1]) for x in seen] if k == "ncu"]
    assert ncu == ["base", "~no_b", "~hang", "base", "~no_b", "~hang"]
    kinds = [(x[0], x[1]) for x in seen if x[1] == "~no_b"]
    assert kinds[0] == ("ncu", "~no_b") and kinds[1] == ("bench", "~no_b")
    # the ablated source was what ran
    assert any(k == "bench" and v == "~no_b" and "B = 0" in src for k, v, src in
               [x for x in seen if x[0] == "bench"])
    # broken-but-timed: positive delta (base 1000 / 900 -> +11.11%), no device delta
    assert ("ABLATE no_b delta=+11.11% ncu_ns=900.0 base_ncu_ns=1000.0 correctness=fail "
            "device_delta=n/a kernel=main_kernel status=ok h=B is the exposed wait") in out
    # no duration rows: n/a with its status, never 0
    assert "ABLATE hang delta=n/a ncu_ns=None base_ncu_ns=1000.0" in out and "status=nodata" in out
    assert "PRICE ablation_noise_floor=" in out and "ABLATE-OK ablations=1 rounds=2 base_ncu_ns=1000.0" in out
    assert (root / "kernel.py").read_text() == BASE
    # the bench is not re-run for a variant that already failed
    assert [x[1] for x in seen if x[0] == "bench"].count("~no_b") == 1


def test_ablation_only_spec_skips_the_lever_table(tmp_path, monkeypatch, capsys):
    """A JSON verdict probe carries ablations and no levers: no lever table,
    no PRICE-FAIL, just the ablation pass; both empty is still refused."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    (root / "notes" / "abl.json").write_text(json.dumps({
        "ablation_rounds": 1, "ablations": [{"name": "no_b", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]}]}))
    (root / "notes" / "empty.json").write_text(json.dumps({"levers": [], "ablations": []}))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setattr(lpr, "run_bench", lambda v, keep_cubin, timeout=None: {
        "correctness": {"pass": True}, "performance": {"device_median_ms": 2.0}})
    monkeypatch.setattr(lpr, "run_price", lambda v, **_: (_ for _ in ()).throw(AssertionError("no roles")))
    monkeypatch.setattr(lpr, "run_ncu_time", _fake_ncu({"base": 1000.0, "~no_b": 1020.0}))
    assert lpr.main(["notes/abl.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE-FAIL" not in out and "LEVER " not in out and "STACK " not in out and "PRICE-OK" not in out
    assert "ABLATE no_b delta=-1.96% ncu_ns=1020.0 base_ncu_ns=1000.0 correctness=pass device_delta=+0.00%" in out
    assert "ABLATE-OK ablations=1 rounds=1" in out
    assert (root / "kernel.py").read_text() == BASE
    assert lpr.main(["notes/empty.json"]) == 0
    assert "PRICE-FAIL reason=no-levers" in capsys.readouterr().out


def test_hung_ablation_costs_one_ncu_timeout_and_nothing_else(tmp_path, monkeypatch, capsys):
    """A deadlocked ablation: ncu times out ONCE, the variant is marked hung —
    no bench, no second round — and its line reads n/a status=hung; the other
    ablations and base are unaffected. A spent pass budget marks what is left
    status=budget without running it."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    (root / "notes" / "abl.json").write_text(json.dumps({"ablation_rounds": 2, "ablations": [
        {"name": "dead", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]},
        {"name": "fine", "edits": [{"old": "C = 3\n", "new": "C = 0\n"}]}]}))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    calls = []
    monkeypatch.setattr(lpr, "run_bench", lambda v, keep_cubin, timeout=None: calls.append(("bench", v)) or
                        {"correctness": {"pass": True}, "performance": {"device_median_ms": 2.0}})

    def ncu(v):
        calls.append(("ncu", v))
        if v == "~dead":
            raise lpr.AblationHung("~dead: ncu did not return")
        return (1000.0 if v == "base" else 800.0, "main_kernel", 8)
    monkeypatch.setattr(lpr, "run_ncu_time", ncu)
    assert lpr.main(["notes/abl.json"]) == 0
    out = capsys.readouterr().out
    assert calls.count(("ncu", "~dead")) == 1 and ("bench", "~dead") not in calls
    assert calls.count(("ncu", "~fine")) == 2 and calls.count(("ncu", "base")) == 2
    assert "ABLATE dead delta=n/a ncu_ns=None base_ncu_ns=1000.0 correctness=skipped device_delta=n/a kernel=none status=hung" in out
    assert "ABLATE fine delta=+25.00%" in out and "status=ok" in out
    assert "ABLATE-OK ablations=1 rounds=2" in out
    # budget: a clock that is already past the budget skips every variant
    (root / "kernel.py").write_text(BASE)
    calls.clear()
    t = iter([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    lpr.time_ablations([{"name": "fine", "edits": [{"old": "C = 3\n", "new": "C = 0\n"}]}],
                       {"~fine": BASE.replace("C = 3", "C = 0")}, 1, str(root / "kernel.py"), BASE,
                       budget_s=5, clock=lambda: next(t))
    out = capsys.readouterr().out
    assert not calls and "ABLATE-FAIL reason=base-not-timed" in out
    assert "skipped: pass budget 5s spent" in out


def test_parse_lever_prices_reads_ablate_rows():
    log = (
        "ABLATE no_b delta=+11.11% ncu_ns=900.0 base_ncu_ns=1000.0 correctness=fail "
        "device_delta=n/a kernel=main_kernel status=ok h=B is the exposed wait\n"
        "ABLATE hang delta=n/a ncu_ns=None base_ncu_ns=1000.0 correctness=nojson "
        "device_delta=n/a kernel=none status=hung h=\n"
        "PRICE ablation_noise_floor=1.50% metric=ncu_ns\n"
        "ABLATE-OK ablations=1 rounds=2 base_ncu_ns=1000.0\n")
    rows, stack, fail = parse_lever_prices(log)
    abl = [r for r in rows if r["kind"] == "ablation"]
    assert fail is None
    assert abl[0] == {"kind": "ablation", "name": "no_b", "delta_pct": 11.11, "ncu_ns": 900.0,
                      "base_ncu_ns": 1000.0, "correctness": "fail", "device_delta_pct": None,
                      "kernel": "main_kernel", "status": "ok", "hypothesis": "B is the exposed wait",
                      "noise_pct": 1.5}
    assert abl[1]["delta_pct"] is None and abl[1]["hypothesis"] == "" and abl[1]["noise_pct"] == 1.5
    assert abl[1]["status"] == "hung"
    # a line without the status field (older tool) still parses
    old = parse_lever_prices("ABLATE x delta=+2.00% ncu_ns=1 base_ncu_ns=1 correctness=pass "
                             "device_delta=n/a kernel=k h=\n")[0][0]
    assert old["status"] == "ok" and old["delta_pct"] == 2.0
    # a demangled kernel name with spaces (a log from before the tool tokenised
    # it) does not unparse the row
    spaced = parse_lever_prices(
        "ABLATE y delta=+3.00% ncu_ns=1 base_ncu_ns=1 correctness=fail device_delta=n/a "
        "kernel=void at::native::elementwise_kernel<128, 4>(int, float *) status=ok h=the h\n")[0][0]
    assert spaced["kernel"] == "void at::native::elementwise_kernel<128, 4>(int, float *)"
    assert spaced["status"] == "ok" and spaced["delta_pct"] == 3.0 and spaced["hypothesis"] == "the h"


def test_ablation_pass_prints_the_kernel_as_one_token(tmp_path, monkeypatch, capsys):
    # ncu reports demangled names with spaces; on the ABLATE line they would
    # end the kernel= field early and the driver would read no row at all
    kpath = tmp_path / "kernel.py"
    kpath.write_text(BASE)
    abl = [{"name": "no_b", "hypothesis": "h", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]}]
    variants = lpr.build_ablation_variants(BASE, abl)
    monkeypatch.setattr(lpr, "run_bench", lambda v, keep_cubin, timeout=None: None)
    monkeypatch.setattr(lpr, "run_ncu_time",
                        lambda v: (1000.0 if v == "base" else 900.0,
                                   "void at::native::elementwise_kernel<128, 4>(int, float *)", 3))
    lpr.time_ablations(abl, variants, 1, str(kpath), BASE)
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("ABLATE no_b "))
    assert " kernel=void_at::native::elementwise_kernel<128,_4>(int,_float_*) status=ok h=h" in line
    from forgekernel.loop.driver import parse_lever_prices as parse
    row = parse(out)[0][0]
    assert row["name"] == "no_b" and row["delta_pct"] == 11.11 and row["status"] == "ok"
    assert lpr.kernel_token(None) == "none" and lpr.kernel_token("  a  b ") == "a_b"


# --- node hygiene ------------------------------------------------------------
# A timed-out probe command used to die alone: subprocess.run(shell=True,
# timeout=) kills the shell (or ncu) and the launch target ncu spawned lived on
# with a deadlocked kernel on the GPU. On the same machine every later
# bench reading then shared the GPU with it (2.3x slow), while ncu — which
# suspends other contexts while it replays — read the kernel unchanged, and the
# run's confirmation archived the inflated number.


def test_sh_timeout_kills_the_whole_process_group(tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setattr(lpr, "ROOT", str(tmp_path))
    monkeypatch.setattr(lpr, "gpu_compute_pids", lambda: [])   # no nvidia-smi here
    monkeypatch.setattr(lpr, "_node_dirty", [])
    pidfile = tmp_path / "child.pid"
    rc, _out, err = lpr.sh(f"sh -c 'echo $$ > {pidfile}; exec sleep 30' & sleep 30", timeout=1)
    assert rc == 124 and "TIMEOUT" in err
    pid = int(pidfile.read_text().strip())
    for _ in range(50):          # the grandchild must be gone, not merely orphaned
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        raise AssertionError("the grandchild survived the timeout (no group kill)")
    assert lpr.node_dirty() is None


def test_sweep_reports_node_dirty_when_the_gpu_stays_held(monkeypatch, capsys):
    monkeypatch.setattr(lpr, "_node_dirty", [])
    monkeypatch.setattr(lpr, "_looks_like_ours", lambda pid: False)   # not ours to kill
    monkeypatch.setattr(lpr, "gpu_compute_pids", lambda: [4242])
    left = lpr.sweep_gpu_orphans("after a probe timed out", wait_s=0.0, sleep=lambda s: None)
    out = capsys.readouterr().out
    assert left == [4242]
    assert "NODE-DIRTY after a probe timed out: pid(s) [4242] still hold the GPU" in out
    assert lpr.node_dirty().startswith("after a probe timed out")
    # released within the wait: clean, nothing recorded
    monkeypatch.setattr(lpr, "_node_dirty", [])
    seq = iter([[4242], []])
    monkeypatch.setattr(lpr, "gpu_compute_pids", lambda: next(seq))
    assert lpr.sweep_gpu_orphans("x", wait_s=5.0, sleep=lambda s: None) == []
    assert "GPU released" in capsys.readouterr().out and lpr.node_dirty() is None
    # nothing on the GPU: silent
    monkeypatch.setattr(lpr, "gpu_compute_pids", lambda: [])
    assert lpr.sweep_gpu_orphans("x") == [] and capsys.readouterr().out == ""


def test_hung_ablation_on_a_dirty_node_fails_the_pass(tmp_path, monkeypatch, capsys):
    """The hung target kept the GPU: no later reading in the pass can be
    trusted — ABLATE-FAIL reason=node-dirty and nothing else runs (the driver
    reruns once)."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    (root / "notes" / "abl.json").write_text(json.dumps({"ablation_rounds": 2, "ablations": [
        {"name": "dead", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]},
        {"name": "fine", "edits": [{"old": "C = 3\n", "new": "C = 0\n"}]}]}))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setattr(lpr, "_node_dirty", [])
    calls = []
    monkeypatch.setattr(lpr, "run_bench", lambda v, keep_cubin, timeout=None: calls.append(("bench", v)) or
                        {"correctness": {"pass": True}, "performance": {"device_median_ms": 2.0}})

    def ncu(v):
        calls.append(("ncu", v))
        if v == "~dead":
            lpr._node_dirty.append("pid(s) [7] still hold the GPU")
            raise lpr.AblationHung("~dead: ncu did not return")
        return (1000.0, "main_kernel", 8)
    monkeypatch.setattr(lpr, "run_ncu_time", ncu)
    assert lpr.main(["notes/abl.json"]) == 0
    out = capsys.readouterr().out
    assert "ABLATE-FAIL reason=node-dirty (pid(s) [7] still hold the GPU)" in out
    assert "ABLATE-OK" not in out
    assert ("ncu", "~fine") not in calls and ("bench", "~dead") not in calls


def test_stack_pricing_stops_on_a_dirty_node(tmp_path, monkeypatch, capsys):
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    (root / "notes" / "levers.json").write_text(json.dumps({"rounds": 2, "levers": [
        {"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}]}))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setattr(lpr, "_node_dirty", [])
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    seen = []

    def bench(v, keep_cubin, timeout=None):
        seen.append(v)
        if v == "+a":   # this bench hung; its sweep found the GPU still held
            lpr._node_dirty.append("pid(s) [9] still hold the GPU")
            return None
        return {"correctness": {"pass": True}, "performance": {"device_median_ms": 1.0}}
    monkeypatch.setattr(lpr, "run_bench", bench)
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE-FAIL reason=node-dirty (pid(s) [9] still hold the GPU)" in out
    assert out.count("PRICE-FAIL") == 1 and "PRICE-OK" not in out
    assert seen == ["base", "+a"]


def test_sweep_treats_an_unavailable_nvidia_smi_as_node_dirty(monkeypatch, capsys):
    """nvidia-smi hung or missing after a timeout is the state a wedged driver
    leaves behind: "cannot tell" must not read as "clean"."""
    monkeypatch.setattr(lpr, "_node_dirty", [])
    monkeypatch.setattr(lpr, "gpu_compute_pids", lambda: None)
    assert lpr.sweep_gpu_orphans("after `ncu` timed out", wait_s=0.0, sleep=lambda s: None) == []
    out = capsys.readouterr().out
    assert "NODE-DIRTY after `ncu` timed out: nvidia-smi unavailable" in out
    assert lpr.node_dirty() and "cannot confirm it is free" in lpr.node_dirty()


def test_role_pass_stops_on_a_dirty_node_before_the_final_table_and_ablations(tmp_path, monkeypatch, capsys):
    """A timed-out role command (stall_price_run) left the GPU held: the timed
    table already on the log stands, but no FINAL table and no ablation pass may
    read the orphan's share of the GPU."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    (root / "notes" / "levers.json").write_text(json.dumps({"rounds": 1, "levers": [
        {"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}],
        "ablations": [{"name": "dead", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]}]}))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setattr(lpr, "_node_dirty", [])
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    monkeypatch.setattr(lpr, "run_bench", _fake_bench({"base": 0.60, "+a": 0.636, "all": 0.636}))
    priced, ncu_calls = [], []

    def price(v, **_):
        priced.append(v)
        if v == "+a":
            lpr._node_dirty.append("pid(s) [11] still hold the GPU")
            return None
        return {"tensor_active": 40.0, "roles": {"spin": {"exposed_ns": 5.0}}}
    monkeypatch.setattr(lpr, "run_price", price)
    monkeypatch.setattr(lpr, "run_ncu_time", lambda v: ncu_calls.append(v) or (1000.0, "k", 8))
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert "PRICE-TIMED" in out and "PRICE-FAIL reason=node-dirty (pid(s) [11] still hold the GPU)" in out
    assert "PRICE-OK" not in out and "ABLATE" not in out and ncu_calls == []
    assert priced == ["base", "+a"]


def test_ablation_bench_on_a_dirty_node_fails_the_pass(tmp_path, monkeypatch, capsys):
    """The correctness bench (not ncu) of an ablation timed out and left the GPU
    held: the pass stops right there, no further round."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    (root / "notes" / "abl.json").write_text(json.dumps({"ablation_rounds": 2, "ablations": [
        {"name": "slow", "edits": [{"old": "B = 2\n", "new": "B = 0\n"}]},
        {"name": "fine", "edits": [{"old": "C = 3\n", "new": "C = 0\n"}]}]}))
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setattr(lpr, "_node_dirty", [])
    calls = []

    def bench(v, keep_cubin, timeout=None):
        calls.append(("bench", v))
        if v == "~slow":
            lpr._node_dirty.append("pid(s) [13] still hold the GPU")
            return None
        return {"correctness": {"pass": True}, "performance": {"device_median_ms": 2.0}}
    monkeypatch.setattr(lpr, "run_bench", bench)
    monkeypatch.setattr(lpr, "run_ncu_time", lambda v: calls.append(("ncu", v)) or (1000.0, "main_kernel", 8))
    assert lpr.main(["notes/abl.json"]) == 0
    out = capsys.readouterr().out
    assert "ABLATE-FAIL reason=node-dirty (pid(s) [13] still hold the GPU)" in out
    assert "ABLATE-OK" not in out and ("ncu", "~fine") not in calls


def test_role_pass_plan_reads_the_job_clock_and_skips_what_it_cannot_cover():
    """The role pass runs under gpu_entry's TASK_TIMEOUT: a --pm is started only
    when the clock left (minus the margin) covers the longest --pm seen so far
    (ROLE_MIN_CALL_S before any), its own timeout is clamped to that clock, and
    an unknown clock (hand run, no TASK_TIMEOUT) leaves the pass unbounded."""
    env = {"TASK_TIMEOUT": "1000"}
    assert lpr.job_seconds_left(0.0, clock=lambda: 400.0, env=env) == 600.0
    assert lpr.job_seconds_left(0.0, clock=lambda: 400.0, env={}) is None
    assert lpr.job_seconds_left(0.0, clock=lambda: 400.0, env={"TASK_TIMEOUT": "x"}) is None
    assert lpr.job_seconds_left(0.0, clock=lambda: 400.0, env={"TASK_TIMEOUT": "0"}) is None
    # 600 s left, 180 margin -> 420 usable: covers a 200 s call, clamped to 420
    assert lpr.role_pass_plan(0.0, [200.0], clock=lambda: 400.0, env=env) == (True, 420, 600.0)
    # the longest call seen (500 s) does not fit: skip
    assert lpr.role_pass_plan(0.0, [200.0, 500.0], clock=lambda: 400.0, env=env) == (False, 0, 600.0)
    # before any call the floor is ROLE_MIN_CALL_S
    assert lpr.role_pass_plan(0.0, [], clock=lambda: 701.0, env=env) == (False, 0, 299.0)
    assert lpr.role_pass_plan(0.0, [], clock=lambda: 600.0, env=env) == (True, 220, 400.0)
    # unknown clock: unbounded, the call keeps its own cap
    assert lpr.role_pass_plan(0.0, [900.0], clock=lambda: 1e9, env={}) == (True, lpr.ROLE_PM_TIMEOUT, None)


def test_main_prints_price_ok_when_the_job_clock_cannot_cover_the_role_pass(tmp_path, monkeypatch, capsys):
    """A role pass that outlives TASK_TIMEOUT dies under SIGKILL with no
    PRICE-OK. Under a clock that cannot cover even one --pm the pass is skipped
    (one line says so, naming the unpriced variants), the PRICE-OK copy is
    printed with the role columns None, and the driver reads it as a table."""
    from forgekernel.loop.driver import _price_log_verdict
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"rounds": 1, "levers": [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}]}))
    monkeypatch.setattr(lpr, "run_bench", _fake_bench({"base": 0.60, "+a": 0.66, "all": 0.66, "all-a": 0.60}))
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    priced = []
    monkeypatch.setattr(lpr, "run_price", lambda v, **kw: priced.append((v, kw.get("timeout"))) or
                        {"tensor_active": 40.0, "roles": {"spin": {"exposed_ns": 100.0}}})
    # a clock that (after the margin) cannot cover ROLE_MIN_CALL_S
    monkeypatch.setenv("TASK_TIMEOUT", str(lpr.ROLE_PASS_MARGIN_S + lpr.ROLE_MIN_CALL_S - 1))
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert priced == []
    assert "PRICE ROLES base skipped: job clock has" in out
    assert "3 variant(s) unpriced for roles: base +a all" in out
    assert out.index("PRICE-TIMED levers=1 rounds=1") < out.index("PRICE-OK levers=1 rounds=1")
    assert "spin_add=None spin_all_minus=None" in out
    assert _price_log_verdict(out) == "PRICE-OK"
    levers, stack, fail = parse_lever_prices(out)
    assert fail is None and levers[0]["name"] == "a" and levers[0]["spin_add_ns"] is None
    assert (root / "kernel.py").read_text() == BASE
    # a clock that covers the pass: every variant priced, the call's timeout clamped
    priced.clear()
    monkeypatch.setenv("TASK_TIMEOUT", "1000")
    assert lpr.main(["notes/levers.json"]) == 0
    out = capsys.readouterr().out
    assert [v for v, _ in priced] == ["base", "+a", "all"]
    assert all(t is not None and t <= 1000 - lpr.ROLE_PASS_MARGIN_S for _, t in priced)
    assert "skipped: job clock" not in out and "PRICE-OK" in out
    # no clock at all: unbounded, the call keeps its own cap
    priced.clear()
    monkeypatch.delenv("TASK_TIMEOUT")
    assert lpr.main(["notes/levers.json"]) == 0
    assert {t for _, t in priced} == {lpr.ROLE_PM_TIMEOUT}


def test_run_bench_scopes_a_group_card_to_its_representative_and_the_table_says_so(monkeypatch):
    """Regression: a group card's pricing job timed every workload of the group for every
    variant x round and was killed at the job clock with no verdict. Every pricing bench
    asks the card for its representative workload only (a card without group support
    ignores the variable); the scope the card reports is printed as PRICE-SCOPE and the
    driver banks it on the stack row."""
    envs = []
    js = {"correctness": {"pass": True}, "performance": {"device_median_ms": 0.03},
          "workload_scope": {"workloads": "representative", "rep": "wl2", "of": 8}}
    monkeypatch.setattr(lpr, "sh", lambda cmd, env=None, timeout=0: envs.append(env) or
                        (0, "=== JSON ===\n" + json.dumps(js), ""))
    monkeypatch.setattr(lpr, "_SCOPE_SEEN", {})
    j = lpr.run_bench("base", keep_cubin=False)
    assert envs[-1] == {lpr.WORKLOAD_SCOPE_ENV: "representative"}
    assert j["workload_scope"]["rep"] == "wl2"
    assert lpr.scope_line() == "PRICE-SCOPE workloads=representative rep=wl2 of=8"
    lpr.run_bench("+x", keep_cubin=True)                       # cubin dump keeps the scope
    assert envs[-1][lpr.WORKLOAD_SCOPE_ENV] == "representative" and envs[-1]["CUTE_DSL_KEEP_CUBIN"] == "1"
    # a single-workload card reports no scope: no PRICE-SCOPE line
    monkeypatch.setattr(lpr, "_SCOPE_SEEN", {})
    js.pop("workload_scope")
    lpr.run_bench("base", keep_cubin=False)
    assert lpr.scope_line() is None
    log = ("LEVER a add=+1.00% remove=-1.00% add_ms=0.1 remove_ms=0.1 class=independent "
           "spin_add=0 spin_all_minus=0\n"
           "STACK all=+1.00% sum_singles=+1.00% coupling=+0.00% all_pos=n/a all_pos_levers=none dropped=none\n"
           "PRICE-SCOPE workloads=representative rep=wl2 of=8\n"
           "PRICE noise_floor=0.50% metric=vs_ref_device sigma=n/a rounds=3 outliers=none\n"
           "PRICE-OK levers=1 rounds=3 base_ms=0.03 all_ms=0.029\n")
    _, stack, fail = parse_lever_prices(log)
    assert fail is None and stack["workload_scope"] == {"workloads": "representative", "rep": "wl2", "of": "8"}
    assert stack["noise_pct"] == 0.5                           # the PRICE line still parses
    _, stack, _ = parse_lever_prices(log.replace("PRICE-SCOPE workloads=representative rep=wl2 of=8\n", ""))
    assert "workload_scope" not in stack


def _meta_card(tmp_path, meta=("wl0", "wl2")):
    p = tmp_path / "eval" / "leaderboard" / "problem.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"group": [{"name": f"wl{i}"} for i in range(3)],
                             **({"meta_workloads": list(meta)} if meta else {})}))
    return tmp_path


def test_workload_flag_scopes_every_bench_role_and_ncu_call_to_that_workload(tmp_path, monkeypatch,
                                                                              capsys):
    """The driver prices each of a group card's meta_workloads in its own job
    (`--workload <name>`); every bench, the role pass and the ablations' NCU clock of that
    job read the named workload, and the table's PRICE-SCOPE names it."""
    monkeypatch.setattr(lpr, "ROOT", str(_meta_card(tmp_path)))
    (tmp_path / "notes").mkdir()
    (tmp_path / "kernel.py").write_text("X = 1\n")
    (tmp_path / "notes" / "levers.json").write_text(json.dumps(
        {"rounds": 1, "roles": True, "levers": [{"name": "a", "edits": [{"old": "X = 1", "new": "X = 2"}]}],
         "ablations": [{"name": "z", "edits": [{"old": "X = 1", "new": "X = 3"}], "hypothesis": "h"}]}))
    envs = []

    def sh(cmd, env=None, timeout=0):
        envs.append((cmd.split()[1] if cmd.startswith("python3") else cmd.split()[0], dict(env or {})))
        if "bench.py" in cmd:
            js = {"correctness": {"pass": True}, "performance": {"device_median_ms": 0.03, "vs_ref_device": 1.0},
                  "workload_scope": {"workloads": "wl2", "rep": "wl2", "of": 3}}
            return 0, "=== JSON ===\n" + json.dumps(js), ""
        return 0, "", ""
    monkeypatch.setattr(lpr, "sh", sh)
    assert lpr.main(["notes/levers.json", "--workload", "wl2"]) == 0
    assert envs and all(e.get(lpr.WORKLOAD_SCOPE_ENV) == "wl2" for _, e in envs)
    assert {c for c, _ in envs} >= {"eval/bench.py", "notes/stall_price_run.py", "ncu"}
    assert "PRICE-SCOPE workloads=wl2 rep=wl2 of=3" in capsys.readouterr().out
    # the next job of the same process starts from the representative again
    envs.clear()
    lpr.main(["notes/levers.json"])
    assert all(e.get(lpr.WORKLOAD_SCOPE_ENV) == "representative" for _, e in envs)


def test_workload_flag_is_refused_on_a_card_without_meta_workloads(tmp_path, monkeypatch, capsys):
    """An older card's runtime ignores a named scope and would time its whole group: the
    tool refuses the name unless the card lists its meta_workloads (and has the entry)."""
    assert lpr.scoped_workload_refused("wl1", root=str(_meta_card(tmp_path / "a"))) is None
    assert "not a workload" in lpr.scoped_workload_refused("wl7", root=str(_meta_card(tmp_path / "b")))
    assert "meta_workloads" in lpr.scoped_workload_refused("wl1", root=str(_meta_card(tmp_path / "c", meta=())))
    assert "bad workload" in lpr.scoped_workload_refused("wl1;rm", root=str(tmp_path))
    # a correctness-only workload (problem.json check_only) is a workload of the card too
    e = _meta_card(tmp_path / "e")
    pj = e / "eval" / "leaderboard" / "problem.json"
    pj.write_text(json.dumps(dict(json.loads(pj.read_text()), check_only=[{"name": "wl9"}])))
    assert lpr.scoped_workload_refused("wl9", root=str(e)) is None
    monkeypatch.setattr(lpr, "ROOT", str(_meta_card(tmp_path / "d", meta=())))
    monkeypatch.setattr(lpr, "sh", lambda *a, **k: pytest.fail("no job work after a refusal"))
    assert lpr.main(["notes/levers.json", "--workload", "wl1"]) == 0
    assert "PRICE-FAIL reason=workload-refused" in capsys.readouterr().out


def test_no_roles_flag_prices_without_the_role_pass_and_still_prints_the_full_table(tmp_path, monkeypatch,
                                                                                     capsys):
    """The driver passes --no-roles on a group card's non-representative workloads (their
    role columns were never read and cost a third of each job): no --pm call, the table still
    ends in PRICE-OK (so the merge counts it), the role columns n/a, the flag is never taken
    for the spec path."""
    root = tmp_path
    (root / "notes").mkdir()
    (root / "kernel.py").write_text(BASE)
    monkeypatch.setattr(lpr, "ROOT", str(root))
    (root / "notes" / "levers.json").write_text(json.dumps(
        {"rounds": 2, "levers": [{"name": "a", "edits": [{"old": "A = 1\n", "new": "A = 10\n"}]}]}))
    monkeypatch.setattr(lpr, "run_bench", _fake_bench({"base": 0.60, "+a": 0.636, "all": 0.636, "all-a": 0.60}))
    monkeypatch.setattr(lpr, "cubin_sha", lambda v: None)
    monkeypatch.setattr(lpr, "run_price", lambda v, **_: (_ for _ in ()).throw(AssertionError("no role pass")))
    assert lpr.main(["notes/levers.json", "--no-roles"]) == 0
    out = capsys.readouterr().out
    assert "PRICE ROLES skipped: --no-roles" in out and "PRICE-OK levers=1" in out
    levers, stack, fail = parse_lever_prices(out)
    assert fail is None and levers[0]["cls"] == "independent" and levers[0]["spin_add_ns"] is None
    assert (root / "kernel.py").read_text() == BASE
