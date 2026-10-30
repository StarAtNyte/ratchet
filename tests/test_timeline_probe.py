"""The timeline probe: markers -> stamps (source rewrite), slab -> per-role
segment account (reduction), machine lines -> dict (parse)."""
from forgekernel.skills.timeline_probe import (
    NROW, NSITE, NSLOT, NWARP, SPAN_BAND, coverage_class, coverage_span, markers,
    parse_timeline, print_account, reduce_slab, split_runs, stamp_source, uneven_spans,
    verdict,
)

TOY = '''"""toy kernel"""
import cutlass.cute as cute


def _seg(site, row=0):
    """timeline marker: no-op here"""
    return None


@cute.kernel
def kern(x):
    _seg("aux.entry")
    for ci in range(128):
        _seg("state.A.top", ci)
        _seg("state.C.wait_kk.begin", ci)
        wait()
        _seg('state.C.wait_kk.end', ci)
        _seg("state.C.newv.drain", ci + 0)
        _seg("state.E.end", ci)
'''


def test_markers_and_stamping_rewrite_the_marker_definition_and_calls():
    assert markers(TOY) == ["aux.entry", "state.A.top", "state.C.wait_kk.begin",
                            "state.C.wait_kk.end", "state.C.newv.drain", "state.E.end"]
    stamped, sites = stamp_source(TOY)
    assert sites == markers(TOY)
    assert "timeline marker: no-op here" not in stamped and "_tl_store(gate" in stamped
    assert "_seg(0, 0)" in stamped and "_seg(1, ci)" in stamped and "_seg(3, ci)" in stamped
    assert "_seg(4, ci + 0)" in stamped
    assert "%clock64" in stamped and "@NSLOT@" not in stamped and "@NONCE@" not in stamped
    assert f"_TL_NSLOT, _TL_NWARP, _TL_NROW, _TL_NSITE = {NSLOT}, {NWARP}, {NROW}, {NSITE}" in stamped
    # the first NSLOT x STRIDE CTAs in LINEAR order are sampled (a grid thin along x
    # still fills every slot), rows beyond the slab are not stored
    assert "cta = bidx + gdx * (bidy + gdy * bidz)" in stamped and "grid_dim()" in stamped
    assert "| (slot // _TL_NSLOT) | (row // _TL_NROW)" in stamped and NROW == 257
    # every stamping gets its own nonce (a source-keyed JIT cache never serves another process's cubin)
    assert stamp_source(TOY)[0] != stamped
    # the stamped file is still python
    compile(stamped, "stamped.py", "exec")
    # waits mode keeps begin/end PAIRS only; a lone `.end` (a segment's end) and the rest become no-ops
    lean, kept = stamp_source(TOY, mode="waits")
    assert kept == ["state.C.wait_kk.begin", "state.C.wait_kk.end"]
    assert "_seg(0, ci)" in lean and "_seg(1, ci)" in lean and lean.count("None\n") == 4
    import pytest
    with pytest.raises(ValueError):
        stamp_source(TOY.replace("def _seg", "def _nope"))
    with pytest.raises(ValueError):
        stamp_source("def _seg(site, row=0):\n    return None\n")


def _slab(sites):
    arr = [[[[0] * NSITE for _ in range(NROW)] for _ in range(NWARP)] for _ in range(NSLOT)]
    idx = {n: i for i, n in enumerate(sites)}
    period = 1000
    for slot in range(2):
        base = 10_000 * (slot + 1)
        for row in range(8):
            t = base + row * period
            # two state warps (4, 5): top, wait begin at +100, end at +400, drain at +700
            for w in (4, 5):
                arr[slot][w][row][idx["state.A.top"]] = t
                arr[slot][w][row][idx["state.C.wait_kk.begin"]] = t + 100
                arr[slot][w][row][idx["state.C.wait_kk.end"]] = t + 400
                arr[slot][w][row][idx["state.C.newv.drain"]] = t + 700
            # one aux warp (12): top at +50, commit at +350 (right before the state passes its wait)
            arr[slot][12][row][idx["aux.a.top"]] = t + 50
            arr[slot][12][row][idx["aux.c.kk.commit"]] = t + 350
        arr[slot][12][128][idx["aux.entry"]] = base - 5
    return arr


SITES = ["aux.entry", "state.A.top", "state.C.wait_kk.begin", "state.C.wait_kk.end",
         "state.C.newv.drain", "aux.a.top", "aux.c.kk.commit"]


def test_reduce_slab_reads_segments_waits_period_and_phase():
    res = reduce_slab(_slab(SITES), SITES)
    assert set(res["roles"]) == {"state", "aux"} and res["slots"] == [0, 1]
    st = res["roles"]["state"]
    assert st["warps"] == 4 and st["order"] == ["state.A.top", "state.C.wait_kk.begin",
                                                "state.C.wait_kk.end", "state.C.newv.drain"]
    segs = {(s["from"], s["to"]): s for s in st["segments"]}
    assert segs[("state.A.top", "state.C.wait_kk.begin")]["median"] == 100
    assert segs[("state.C.wait_kk.begin", "state.C.wait_kk.end")]["kind"] == "wait"
    assert segs[("state.C.wait_kk.begin", "state.C.wait_kk.end")]["median"] == 300
    assert segs[("state.C.newv.drain", "state.A.top (next)")]["kind"] == "wrap"
    assert segs[("state.C.newv.drain", "state.A.top (next)")]["median"] == 300
    assert st["period"] == 1000 and st["iqr"] == 0 and st["wait"] == 300 and st["busy"] == 400
    assert st["wrap"] == 300 and st["busy"] + st["wait"] + st["wrap"] == st["period"]
    aux = res["roles"]["aux"]
    assert aux["order"] == ["aux.a.top", "aux.c.kk.commit"] and aux["period"] == 1000
    # a role spread over two warpgroups is reported per warpgroup
    arr = _slab(SITES)
    for slot in range(2):
        arr[slot][9] = [row[:] for row in arr[slot][5]]       # a state warp in warpgroup 2
        arr[slot][5] = [[0] * NSITE for _ in range(NROW)]
    split = reduce_slab(arr, SITES)
    assert set(split["roles"]) == {"state@wg1", "state@wg2", "aux"}
    assert split["roles"]["state@wg1"]["warps"] == 2 and split["roles"]["state@wg2"]["period"] == 1000
    # phase: offsets from the row's earliest stamp (the state top)
    assert res["phase"]["aux"]["aux.c.kk.commit"] == 350
    assert res["phase"]["state"]["state.C.wait_kk.end"] == 400
    assert res["mono_bad"] == 0 and res["mono_tot"] > 0
    assert res["restarts"] == 0 and res["restart_pairs"] > 0
    assert res["spans"] and res["spans"][0] == 7 * 1000 + 700 + 5
    assert res["slot_roles"] == [["aux", "state"], ["aux", "state"]]
    assert coverage_span(res) == max(res["spans"]) and uneven_spans(res) is None


def test_print_and_parse_round_trip(capsys):
    res = reduce_slab(_slab(SITES), SITES)
    print_account(res, mode="all", numeric="identical", perturbation=1.031, span_ratio=0.98,
                  sm_mhz=1590.0, base_ms=0.5255, stamped_ms=0.5418, sm_src="sampled")
    print("TIMELINE-WARN perturbation 1.0310 > 1.03 even in mode=all — read the table for its structure")
    print("TIMELINE-OK mode=all")
    out = capsys.readouterr().out
    tl = parse_timeline(out)
    assert tl["ok"] and tl["mode"] == "all" and tl["numeric"] == "identical" and tl["sm_src"] == "sampled"
    assert tl["perturbation"] == 1.031 and tl["span_ratio"] == 0.98 and tl["sm_mhz"] == 1590
    assert tl["restarts"] == 0 and tl["restart_pairs"] == res["restart_pairs"] > 0
    # an older log's field name still parses into the same keys
    assert parse_timeline(out.replace("restarts=0/", "rows_bad=0/"))["restart_pairs"] == res["restart_pairs"]
    assert tl["warn"] == ["perturbation 1.0310 > 1.03 even in mode=all — read the table for its structure"]
    # a line without the trailing source field (older logs) still parses
    assert parse_timeline(out.replace(" sm_src=sampled", ""))["sm_src"] == "default"
    assert tl["roles"]["state"]["period"] == 1000 and tl["roles"]["state"]["wait"] == 300
    assert tl["roles"]["state"]["wrap"] == 300
    assert parse_timeline(out.replace(" wrap=300", ""))["roles"]["state"]["wrap"] is None
    wait = next(s for s in tl["roles"]["state"]["segments"] if s["kind"] == "wait")
    assert (wait["from"], wait["to"], wait["median"]) == ("state.C.wait_kk.begin", "state.C.wait_kk.end", 300)
    wrap = next(s for s in tl["roles"]["state"]["segments"] if s["kind"] == "wrap")
    assert wrap["to"] == "state.A.top (next)"
    assert tl["phase"]["aux"]["aux.c.kk.commit"] == 350
    # a failed run parses closed
    assert parse_timeline("TIMELINE-FAIL reason=markers (no def)\n")["fail"].startswith("markers")
    assert parse_timeline("nothing here") is None
    two = out + out.replace("mode=all", "mode=waits").replace("TIMELINE-OK mode=all", "TIMELINE-OK mode=waits")
    assert parse_timeline(two)["mode"] == "waits"


def _persistent_slab(sites, first=8, last=5, period=1000, gap=3000):
    """A persistent kernel: every sampled CTA walks TWO work items, the row
    counter (pass index) restarting at 0 for the second. The second item is
    shorter, so rows 0..last-1 hold it (fresh) and rows last..first-1 still
    hold the first item (older stamps)."""
    arr = [[[[0] * NSITE for _ in range(NROW)] for _ in range(NWARP)] for _ in range(NSLOT)]
    idx = {n: i for i, n in enumerate(sites)}
    for slot in range(2):
        base = 10_000 * (slot + 1)
        t_second = base + first * period + gap
        for row in range(first):
            t = base + row * period
            if row < last:
                t = t_second + row * period
            for w in (4, 5):
                arr[slot][w][row][idx["state.A.top"]] = t
                arr[slot][w][row][idx["state.C.wait_kk.begin"]] = t + 100
                arr[slot][w][row][idx["state.C.wait_kk.end"]] = t + 400
                arr[slot][w][row][idx["state.C.newv.drain"]] = t + 700
    return arr


def test_split_runs_breaks_at_a_row_counter_restart_and_at_a_row_gap():
    sites = SITES
    arr = _persistent_slab(sites)
    idx = {n: i for i, n in enumerate(sites)}
    order = [idx["state.A.top"], idx["state.C.wait_kk.begin"], idx["state.C.wait_kk.end"],
             idx["state.C.newv.drain"]]
    runs, restarts, pairs = split_runs(arr[0][4], list(range(8)), order)
    assert runs == [[0, 1, 2, 3, 4], [5, 6, 7]] and restarts == 1 and pairs == 7
    # site order does not matter to the split (row min / max decide)
    assert split_runs(arr[0][4], list(range(8)), order[::-1])[0] == runs
    # a gap in the row index starts a new run without counting as a restart
    runs, restarts, pairs = split_runs(arr[0][4], [0, 1, 2, 5, 6, 7], order)
    assert runs == [[0, 1, 2], [5, 6, 7]] and restarts == 0 and pairs == 4


def test_persistent_kernel_rows_are_read_per_run_never_across_the_restart():
    # first item 16 passes, second 10: rows 0..9 fresh, rows 10..15 the older item
    res = reduce_slab(_persistent_slab(SITES, first=16, last=10), SITES)
    st = res["roles"]["state"]
    # one restart per warp (2 slots x 2 warps), reported, not a fault
    assert res["restarts"] == 4 and res["restart_pairs"] == 4 * 15
    assert res["mono_bad"] == 0
    segs = {(s["from"], s["to"]): s for s in st["segments"]}
    # the wrap never pairs row 9 (fresh) with row 10 (the older item): every wrap
    # is one period's 300, and the period is the true per-pass period
    wrap = segs[("state.C.newv.drain", "state.A.top (next)")]
    assert (wrap["median"], wrap["p25"], wrap["p75"]) == (300, 300, 300)
    assert st["period"] == 1000 and st["iqr"] == 0
    # steady rows from both runs, each trimmed by 2 at both ends: (10 - 4) + (6 - 4)
    assert res["rows"] == 8 and st["rows"] == 8
    # wraps: 5 inside the 6-row run + 1 inside the 2-row run, per warp
    assert wrap["n"] == 4 * (5 + 1)
    # the slot span is the union of what SURVIVED: the first item's row 10 (its
    # rows 0..9 were overwritten) -> the second item's last stamp
    assert res["spans"][0] == (16 * 1000 + 3000 + 9 * 1000 + 700) - 10 * 1000
    reasons, warns = verdict(res, numeric="identical", perturbation=1.01, span_ratio=0.98,
                             sm_src="sampled", mode="all", sm_mhz=1590.0)
    assert reasons == [] and any(w.startswith("row-counter restarts 4/60") for w in warns)


def test_persistent_kernel_longer_items_keep_every_row_in_one_run():
    # the second item as long as the first: every row overwritten, one run, no restart
    res = reduce_slab(_persistent_slab(SITES, first=8, last=8), SITES)
    assert res["restarts"] == 0 and res["roles"]["state"]["period"] == 1000
    assert res["rows"] == 4


def test_second_launch_is_told_by_coverage_not_by_the_restart_count():
    # slot 1: rows 0..3 overwritten by a LATER launch of another shape — the
    # restart shows, and the union span of that slot runs across the host gap
    arr = _slab(SITES)
    for w in range(NWARP):
        for row in range(4):
            for i in range(NSITE):
                if arr[1][w][row][i]:
                    arr[1][w][row][i] += 5_000_000
    res = reduce_slab(arr, SITES)
    assert res["restarts"] == 3 and res["restart_pairs"] > res["restarts"]
    assert coverage_span(res) == max(res["spans"]) > 5_000_000
    assert uneven_spans(res) is not None
    device_cycles = 8 * 1000
    ratio = coverage_span(res) / device_cycles
    reasons, warns = verdict(res, numeric="identical", perturbation=1.0, span_ratio=ratio,
                             sm_src="sampled", mode="all", sm_mhz=1590.0)
    assert any(r.startswith("span_ratio") and "more than one launch" in r for r in reasons)
    assert any("more than 2x across slots of the same roles" in w for w in warns)
    # with an unsampled clock the coverage is advisory, the table stands
    reasons, warns = verdict(res, numeric="identical", perturbation=1.0, span_ratio=ratio,
                             sm_src="default", mode="all", sm_mhz=1590.0)
    assert reasons == [] and any("coverage unverified" in w for w in warns)


def test_a_short_second_kernel_in_other_slots_does_not_drag_the_coverage_down():
    # slots 0-1: the main (persistent) kernel's CTAs; slots 2-7: CTAs of a
    # combine kernel launched after it (entry/exit stamps in row 0 only) —
    # a grid of [n_seq, 1, k] whose first CTAs land in the same slots
    sites = SITES + ["combine.entry", "combine.exit"]
    arr = _slab(sites)
    idx = {n: i for i, n in enumerate(sites)}
    for slot in range(2, NSLOT):
        for w in range(8):
            arr[slot][w][0][idx["combine.entry"]] = 200_000 + slot
            arr[slot][w][0][idx["combine.exit"]] = 200_000 + slot + 4_000
    res = reduce_slab(arr, sites)
    assert res["slots"] == list(range(NSLOT))
    assert res["slot_roles"][0] == ["aux", "state"]
    assert res["slot_roles"][2] == ["combine@wg0", "combine@wg1"]
    # the longest CTA (a main-kernel slot) is the coverage reading; the combine
    # slots (4000 cycles) are neither the median nor a 2x-uneven finding
    assert coverage_span(res) == 7 * 1000 + 700 + 5
    assert uneven_spans(res) is None
    assert res["roles"]["combine@wg0"]["period"] is None
    assert res["restarts"] == 0
    reasons, warns = verdict(res, numeric="identical", perturbation=1.0, span_ratio=0.99,
                             sm_src="sampled", mode="all", sm_mhz=1590.0)
    assert reasons == [] and warns == []
    assert SPAN_BAND == (0.75, 1.25)


def test_short_coverage_is_partial_not_a_failure():
    """A span BELOW the band is a real table that does not reach the whole
    device call — the multi-kernel case (GDN bwd: 16 of 17 probes read
    0.62 on a tree whose dominant kernel is 69% of device time, every one of
    them carrying a complete per-role table). It must not read as a failure:
    the probe did its job, and the driver hands the table over."""
    res = reduce_slab(_slab(SITES), SITES)
    reasons, warns = verdict(res, numeric="identical", perturbation=1.0, span_ratio=0.62,
                             sm_src="sampled", mode="all", sm_mhz=1590.0)
    assert reasons == []
    assert any(w.startswith("span_ratio 0.620 below") and "not in this table" in w for w in warns)
    assert coverage_class(res, reasons, span_ratio=0.62, sm_src="sampled") == "partial"
    assert coverage_class(res, reasons, span_ratio=0.98, sm_src="sampled") == "full"
    # above the band stays a FAILURE: the slab holds more than one launch, so
    # the segments mix trees — a wrong table, which is worse than no table
    reasons_hi, _ = verdict(res, numeric="identical", perturbation=1.0, span_ratio=1.9,
                            sm_src="sampled", mode="all", sm_mhz=1590.0)
    assert any(r.startswith("span_ratio") and "more than one launch" in r for r in reasons_hi)
    # a failure has no table to classify; nothing measured is not "short"
    assert coverage_class(res, ["no-stamps-read"], span_ratio=0.62, sm_src="sampled") == "full"
    assert coverage_class(res, [], span_ratio=None, sm_src="sampled") == "full"
    # an unsampled clock is uncertain by its own error (10-15%), not by a factor
    # of two: 0.62 under it is still short and is NOT handed over as a whole-call
    # account — partial, beside the "coverage unverified" warning
    assert coverage_class(res, [], span_ratio=0.62, sm_src="default") == "partial"
    assert coverage_class(res, [], span_ratio=0.98, sm_src="default") == "full"
    _, warns_unsampled = verdict(res, numeric="identical", perturbation=1.0, span_ratio=0.62,
                                 sm_src="default", mode="all", sm_mhz=1590.0)
    assert any("coverage unverified" in w for w in warns_unsampled)


def test_print_parse_carries_the_coverage_class(capsys):
    res = reduce_slab(_slab(SITES), SITES)
    print_account(res, mode="all", numeric="identical", perturbation=1.0, span_ratio=0.62,
                  sm_mhz=1590.0, base_ms=0.8255, stamped_ms=0.8421, sm_src="sampled")
    print("TIMELINE-COVERAGE partial")
    print("TIMELINE-OK mode=all")
    tl = parse_timeline(capsys.readouterr().out)
    assert tl["ok"] and tl["coverage"] == "partial" and tl["usable"] is True
    assert tl["roles"]["state"]["wait"] == 300          # the table came through whole
    # a full-coverage log, and a log from before the line existed
    print_account(res, mode="all", numeric="identical", perturbation=1.0, span_ratio=0.98,
                  sm_mhz=1590.0, base_ms=0.8255, stamped_ms=0.8421, sm_src="sampled")
    print("TIMELINE-COVERAGE full")
    print("TIMELINE-OK mode=all")
    assert parse_timeline(capsys.readouterr().out)["coverage"] == "full"
    print_account(res, mode="all", numeric="identical", perturbation=1.0, span_ratio=0.98,
                  sm_mhz=1590.0, base_ms=0.8255, stamped_ms=0.8421, sm_src="sampled")
    print("TIMELINE-OK mode=all")
    old = parse_timeline(capsys.readouterr().out)
    assert old["coverage"] == "unknown" and old["usable"] and old["ok"]
    # nothing measured: unusable with or without a reason line
    assert parse_timeline("TIMELINE-FAIL reason=markers (no def)\n")["usable"] is False
    assert parse_timeline("TIMELINE-FAIL reason=markers (no def)\n")["coverage"] is None


# --- card fence self-check ------------------------------------------------------
# A GDN card's fence, verbatim (a sealed reference behind a job-env key
# has to fence `os.environ`); its `os\s*\.\s*environ` alternative has no word
# boundary, so `_tl_os.environ` — the preamble's own spelling until 0318c147 —
# was a forbidden op, the stamped bench exited before importing the kernel, and
# every timeline of the run read no-slab-dumped.
GDN_FENCE_BENCH = '''
import re, sys, json

_FORBIDDEN = re.compile(
    r"ref_blob|KDA_REF|kda_ref|kda_gdn|gdn_core|gdn_runtime|gdn_small|gdn_stitch"
    r"|gdn_tile_scheduler|keystream|unseal|decrypt|hashlib|tarfile|tempfile|mkdtemp"
    r"|importlib|linecache|getsource|\\bopen\\s*\\(|os\\s*\\.\\s*environ|/tmp\\b|fkref"
    r"|load_library|ExtensionFileLoader|cpp_extension|load_inline"
    r"|\\bctypes\\b|\\bsubprocess\\b|\\bnvcc\\b|cuobjdump"
    r"|torch\\s*\\.\\s*(matmul|mm|bmm|einsum|addmm|baddbmm|exp|expm1|log|log1p|log2"
    r"|sigmoid|softplus|tanh|cumprod|cumsum|logcumsumexp)\\b"
    r"|\\.(matmul|bmm|einsum|addmm|baddbmm|exp|expm1|log|log1p|log2|sigmoid|softplus"
    r"|tanh|cumprod|cumsum)\\s*\\("
    r"|nn\\.functional\\.(linear|softplus|sigmoid|silu)"
    r"|fla\\b|flash_linear_attention|chunk_gated_delta|flashinfer|\\btriton\\b|cublas|cudnn",
)


def _forbidden_op(kernel_src: str) -> str | None:
    stripped = "\\n".join(line.split("#", 1)[0] for line in kernel_src.splitlines())
    m = _FORBIDDEN.search(stripped)
    return m.group(0) if m else None
'''


def test_card_fence_is_read_from_the_bench_source_without_importing_it():
    from forgekernel.skills.timeline_probe import card_fence, preamble_fence_hit, stamped_preamble
    fence = card_fence(GDN_FENCE_BENCH)
    assert fence is not None
    # the pre-0318c147 preamble spelling tripped this fence; the current one does not
    assert fence.search("_TL_STRIDE = int(_tl_os.environ.get('TL_STRIDE', '1'))").group(0) == "os.environ"
    assert "_tl_os.environ" not in stamped_preamble() and "_tl_os.getenv" in stamped_preamble()
    assert preamble_fence_hit(GDN_FENCE_BENCH) is None
    # a card without a literal fence, or without one at all, has nothing to check
    assert card_fence("import re\n") is None
    assert card_fence("_FORBIDDEN = re.compile(build_pattern())\n") is None
    assert preamble_fence_hit("") is None


def test_a_fence_token_the_preamble_carries_is_named_before_any_job_is_bought():
    from forgekernel.skills.timeline_probe import card_fence, preamble_fence_hit
    # a card that fences `atexit` (the preamble registers its slab dump with it)
    bench = GDN_FENCE_BENCH.replace("|cublas|cudnn", "|cublas|cudnn|\\batexit\\b")
    assert preamble_fence_hit(bench) == "atexit"
    # a token only in a preamble COMMENT is not an op (the card strips comments too)
    bench_c = GDN_FENCE_BENCH.replace("|cublas|cudnn", "|cublas|cudnn|driver-injected")
    assert preamble_fence_hit(bench_c) is None
    # the word-boundary spelling the card should use never sees `_tl_os`
    bounded = GDN_FENCE_BENCH.replace(r"os\s*\.\s*environ", r"\bos\s*\.\s*environ")
    assert bounded != GDN_FENCE_BENCH and preamble_fence_hit(bounded) is None
    assert card_fence(bounded).search("_TL_STRIDE = int(_tl_os.environ.get('TL_STRIDE', '1'))") is None


def test_a_stamped_bench_refusal_is_reported_as_the_benchs_verdict_not_a_missing_slab(tmp_path, monkeypatch, capsys):
    """The card fenced the stamped tree (pass=False, rc 0, kernel never
    imported): the reason line carries the bench's own error text, and the
    driver reads it through parse_timeline. no-slab-dumped is reserved for a
    bench that RAN and left no slab."""
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    monkeypatch.setattr(tp, "run_bench", lambda label, env=None: {
        "correctness": {"pass": True, "max_diff": 0.0},
        "performance": {"device_median_ms": 0.5}})
    refused = {"correctness": {"pass": False, "max_diff": None,
                               "error": "forbidden op in kernel: os.environ"},
               "performance": None}
    monkeypatch.setattr(tp, "run_bench_sampling_clock",
                        lambda label, env=None, timeout=2400: (refused, [], 0))
    assert tp.main(["--mode", "waits"]) == 0
    out = capsys.readouterr().out
    assert "TIMELINE-FAIL reason=stamped-bench-failed mode=waits detail=forbidden op in kernel: os.environ" in out
    assert "no-slab-dumped" not in out
    tl = parse_timeline(out)
    assert tl["ok"] is False and tl["usable"] is False
    assert tl["fail"].startswith("stamped-bench-failed mode=waits detail=forbidden op in kernel: os.environ")
    # the kernel file is restored after the stamped run
    assert (tmp_path / "kernel.py").read_text(encoding="utf-8") == TOY
    # a bench that produced no JSON at all names its rc
    monkeypatch.setattr(tp, "run_bench_sampling_clock",
                        lambda label, env=None, timeout=2400: (None, [], 137))
    tp.main(["--mode", "waits"])
    assert "reason=stamped-bench-failed mode=waits detail=no JSON verdict from the stamped bench (rc=137)" \
        in capsys.readouterr().out


def test_stamped_bench_detail_is_one_bounded_line():
    from forgekernel.skills.timeline_probe import stamped_bench_detail
    assert stamped_bench_detail({"pass": False, "error": "forbidden op\nin kernel:   os.environ"}) \
        == "forbidden op in kernel: os.environ"
    assert stamped_bench_detail({"pass": False}) == "pass=False (no error text)"
    assert stamped_bench_detail(None) == "pass=None (no error text)"
    long = stamped_bench_detail({"pass": False, "error": "x" * 500})
    assert len(long) == 201 and long.endswith("…")


# --- a site name the source text cannot read ---------------------------------
# Observed in a past run (D1.4): six markers in an unrolled loop spelled
# `_seg("wg0.segA.ld%d.begin" % seg, gt)`, then `_seg(_SEGA_LD_B[seg], gt)`. The
# site table is built from the source text, so neither registered a site — and
# left in place the stamped `_seg(site: int, row)` got a str and died at trace
# time. Three trees' timelines were lost, each reported as
# "stamped-build-fails-correctness; no-stamps-read" (an all-zero slab from the
# exit-time dump), and the bench's reason never reached the log.
TOY_COMPUTED = TOY.replace(
    '        _seg("state.E.end", ci)\n',
    '        for k in range(4):\n'
    '            _seg("state.D.ld%d.begin" % k, ci)\n'
    '            _seg(NAMES[k], ci)\n'
    '            _seg(f"state.D.ew{k}.end", ci)\n'
    '        _seg("state.E.end", ci)\n')
DROPPED = ['_seg("state.D.ld%d.begin" % k, ci)', '_seg(NAMES[k], ci)', '_seg(f"state.D.ew{k}.end", ci)']


def test_a_non_literal_site_name_is_a_no_op_in_the_stamped_arm_and_is_named():
    import re
    import pytest
    from forgekernel.skills.timeline_probe import unparsed_markers
    assert TOY_COMPUTED != TOY
    assert markers(TOY_COMPUTED) == markers(TOY)
    assert unparsed_markers(TOY_COMPUTED) == DROPPED
    assert unparsed_markers(TOY) == []
    stamped, sites = stamp_source(TOY_COMPUTED)
    assert sites == markers(TOY)
    # the literal calls are stamped as before; the computed ones are gone, not
    # left for the stamp to choke on
    assert "_seg(1, ci)" in stamped and "_seg(5, ci)" in stamped
    assert "% k" not in stamped and "NAMES[k]" not in stamped and 'f"state.D' not in stamped
    assert re.search(r"(?<!def )_seg\(\s*(?!\d)", stamped) is None
    compile(stamped, "stamped.py", "exec")
    # a call the probe can neither read nor neutralise (nesting deeper than one
    # level) fails the stamping closed, by name
    with pytest.raises(ValueError, match=r"could not read or neutralise: _seg\(\"state.A.top\", f\(g\(ci\)\)\)"):
        stamp_source(TOY.replace('_seg("state.A.top", ci)', '_seg("state.A.top", f(g(ci)))'))


# `_seg(` is also the tail of a helper's name. A DSA kernel called
# `_qk_seg(mma, tcA, tcB, acc,` broken across lines; a self-evolution kernel
# defined `_mask_seg(acc_mn, …)`. Read without a left boundary, the first is a
# residual that fails the stamping closed and the second is rewritten to
# `def _maskNone:` — a stamped source that does not compile (30 of the 453
# kernel versions in earlier runs' histories, all fine under the boundary).
TOY_HELPERS = TOY.replace(
    '@cute.kernel\n',
    'def _mask_seg(acc, s0, nr, nc):\n'
    '    return acc\n\n\n'
    'def _qk_seg(mma, acc, a, b):\n'
    '    return acc\n\n\n'
    '@cute.kernel\n').replace(
    '        _seg("state.E.end", ci)\n',
    '        _seg("state.E.end", ci)\n'
    '        acc = _mask_seg(acc, ci, NR, NC)\n'
    '        acc = _qk_seg(mma, acc,\n'
    '                      view(sA, ci), view(sB, ci))\n')


def test_a_helper_whose_name_ends_in_seg_is_neither_a_marker_nor_a_residual():
    from forgekernel.skills.timeline_probe import unparsed_markers
    assert TOY_HELPERS != TOY
    assert markers(TOY_HELPERS) == markers(TOY)
    assert unparsed_markers(TOY_HELPERS) == []
    stamped, sites = stamp_source(TOY_HELPERS)
    assert sites == markers(TOY)
    assert "def _mask_seg(acc, s0, nr, nc):" in stamped
    assert "acc = _mask_seg(acc, ci, NR, NC)" in stamped
    assert "acc = _qk_seg(mma, acc,\n                      view(sA, ci), view(sB, ci))" in stamped
    assert "_seg(5, ci)" in stamped and "None" not in stamped.split("# ---- end timeline probe")[1]
    compile(stamped, "stamped.py", "exec")
    # a method of that name is not the module-level marker either
    assert markers('x._seg("a.b.c", i)\n') == [] and unparsed_markers('x._seg(name, i)\n') == []


class _FakeNumpy:
    """Stand-in for the `numpy` the probe imports lazily on its slab path (the
    suite carries no numpy): a slab travels as the nested Python list the
    reduction tests build, and `.any()` / `.reshape(...).tolist()` are all the
    probe asks of it. `load` of a path `save` never wrote raises, as numpy
    would on a file that is not a slab."""

    def __init__(self):
        self.saved = {}

    def save(self, path, arr):
        self.saved[str(path)] = arr
        with open(path, "wb") as fh:
            fh.write(b"NUMPY")

    def load(self, path):
        if str(path) not in self.saved:
            raise ValueError("not a slab")
        return _FakeArray(self.saved[str(path)])


class _FakeArray:
    def __init__(self, nested):
        self.nested = nested

    def any(self):
        return any(v for slot in self.nested for w in slot for row in w for v in row)

    def reshape(self, *shape):
        return self

    def tolist(self):
        return self.nested


def _zero_slab():
    return [[[[0] * NSITE for _ in range(NROW)] for _ in range(NWARP)] for _ in range(NSLOT)]


def _wait_slab(rows=10, warps=(4, 5)):
    """A slab with stamps for the two wait sites of TOY in mode=waits."""
    arr = _zero_slab()
    for row in range(rows):
        t = 10_000 + row * 1000
        for w in warps:
            arr[0][w][row][0] = t + 100
            arr[0][w][row][1] = t + 400
    return arr


def _mock_benches(tp, monkeypatch, *, stamped, slab):
    """Base bench passes; the stamped bench returns ``stamped`` and leaves
    ``slab`` (a nested list, or None for no file) at TL_SLAB_OUT. Returns the
    fake numpy the probe will import."""
    import sys
    fake = _FakeNumpy()
    monkeypatch.setitem(sys.modules, "numpy", fake)
    monkeypatch.setattr(tp, "run_bench", lambda label, env=None: {
        "correctness": {"pass": True, "max_diff": 0.0},
        "performance": {"device_median_ms": 0.5}})

    def stamped_bench(label, env=None, timeout=2400):
        if slab is not None:
            fake.save(env["TL_SLAB_OUT"], slab)
        return stamped, [], 0
    monkeypatch.setattr(tp, "run_bench_sampling_clock", stamped_bench)
    return fake


def test_the_dropped_calls_are_named_where_the_driver_reads_warnings(tmp_path, monkeypatch, capsys):
    """End to end: a tree with computed site names still yields its table, and
    the WARN naming the dropped calls lands AFTER the TIMELINE line — the only
    place parse_timeline reads warnings from — so the student and the teacher
    see it in the rendered table, not only in the job log."""
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY_COMPUTED, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    _mock_benches(tp, monkeypatch,
                  stamped={"correctness": {"pass": True, "max_diff": 0.0},
                           "performance": {"device_median_ms": 0.5}},
                  slab=_wait_slab())
    assert tp.main(["--mode", "waits"]) == 0
    out = capsys.readouterr().out
    assert "TIMELINE-DROPPED 3 " + " | ".join(DROPPED) in out
    assert "TIMELINE-SITES 2 0:state.C.wait_kk.begin 1:state.C.wait_kk.end" in out
    tl = parse_timeline(out)
    assert tl["usable"] and tl["numeric"] == "identical" and set(tl["roles"]) == {"state"}
    named = [w for w in tl["warn"] if w.startswith("3 marker call(s) carry a site name that is not a string literal")]
    assert len(named) == 1 and "; ".join(DROPPED) in named[0] and "const_expr(k == 0)" in named[0]
    assert (tmp_path / "kernel.py").read_text(encoding="utf-8") == TOY_COMPUTED


def test_an_all_zero_slab_is_the_benchs_failure_not_a_numeric_difference(tmp_path, monkeypatch, capsys):
    """The slab is allocated at import and dumped at exit, so a stamped bench
    that died at trace time leaves an all-zero slab. That is the bench's own
    verdict (its error text is the one line that says what to fix), not
    "numeric differs, no stamps read"."""
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    died = {"correctness": {"pass": False, "max_diff": None,
                            "error": "TypeError: unsupported operand type(s) for +: 'Int32' and 'str'"},
            "performance": None}
    _mock_benches(tp, monkeypatch, stamped=died, slab=_zero_slab())
    assert tp.main(["--mode", "waits"]) == 0
    out = capsys.readouterr().out
    assert ("TIMELINE-FAIL reason=stamped-bench-failed mode=waits "
            "detail=TypeError: unsupported operand type(s) for +: 'Int32' and 'str'") in out
    assert "stamped-build-fails-correctness" not in out and "no-stamps-read" not in out
    tl = parse_timeline(out)
    assert tl["ok"] is False and tl["usable"] is False and tl["fail"].startswith("stamped-bench-failed")
    # a stamped kernel that RAN (stamps in the slab) and failed the numerics is
    # still the numeric verdict — now with the bench's reason beside it
    ran = {"correctness": {"pass": False, "max_diff": 0.5, "error": "max_diff 0.5 > atol 1e-2"},
           "performance": {"device_median_ms": 0.52}}
    _mock_benches(tp, monkeypatch, stamped=ran, slab=_wait_slab())
    tp.main(["--mode", "waits"])
    out = capsys.readouterr().out
    assert "TIMELINE-FAIL reason=stamped-build-fails-correctness (max_diff 0.5 > atol 1e-2)" in out
    assert parse_timeline(out)["fail"].startswith("stamped-build-fails-correctness (max_diff 0.5")


def test_slab_has_stamps_reads_missing_zero_and_stamped_files(tmp_path, monkeypatch):
    import sys
    from forgekernel.skills.timeline_probe import slab_has_stamps
    fake = _FakeNumpy()
    monkeypatch.setitem(sys.modules, "numpy", fake)
    assert slab_has_stamps(str(tmp_path / "absent.npy")) is False
    fake.save(str(tmp_path / "zero.npy"), _zero_slab())
    assert slab_has_stamps(str(tmp_path / "zero.npy")) is False
    fake.save(str(tmp_path / "stamped.npy"), _wait_slab(rows=1))
    assert slab_has_stamps(str(tmp_path / "stamped.npy")) is True
    (tmp_path / "junk.npy").write_bytes(b"not a slab")
    assert slab_has_stamps(str(tmp_path / "junk.npy")) is False


def test_dry_run_checks_the_stamped_source_with_builtin_compile(tmp_path, monkeypatch, capsys):
    """--dry compiles the stamped source in-process. It used the stdlib byte-compile
    module, whose name is on the closed-source guard's forbid list — and this file is
    seeded into notes/, which the guard scans before every job."""
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    assert tp.main(["--mode", "waits", "--dry"]) == 0
    assert "[dry] stamped source compiles" in capsys.readouterr().out
    # a stamped source that does not parse is reported, not raised
    monkeypatch.setattr(tp, "stamp_source", lambda src, mode: ("def broken(:\n", ["s0"]))
    assert tp.main(["--mode", "waits", "--dry"]) == 0
    assert "TIMELINE-FAIL reason=stamped-source-does-not-compile" in capsys.readouterr().out
    with open(tp.__file__, encoding="utf-8") as fh:
        assert "py" + "_compile" not in fh.read()


# --- one launch per table -------------------------------------------------------
# The stamped kernel writes one slab that is dumped at exit, and a launch only
# overwrites the cells it reaches. On a fused_attn_rope_cast run the bench
# launched 7 correctness draws and 3 perf workloads (a rescale path only `sharp`
# takes, empty rows only `edge` has): stale cells of earlier launches stayed in the
# slab and every tree read span_ratio 200-3000 ("the slab holds more than one
# launch"). Each stamp now carries its launch (%gridid, high 16 bits), only the
# last launch is read, and the table comes from the card's campaign launch target.
WAIT_SITES = ["state.C.wait_kk.begin", "state.C.wait_kk.end"]


def _tag(tag, clock):
    """A stamp as the preamble writes it — launch tag << 48 | clock — as the int64
    bit pattern the slab holds (a tag with bit 15 set is a negative cell)."""
    from forgekernel.skills.timeline_probe import TAG_SHIFT
    v = (tag << TAG_SHIFT) | clock
    return v - (1 << 64) if v >= (1 << 63) else v


def _tagged_wait_slab(tag, t0, rows=10, warps=(4, 5), row_step=1000, arr=None):
    """TOY's two wait sites (mode=waits) stamped by launch ``tag`` from ``t0``,
    written over ``arr`` the way a later launch overwrites the cells it reaches."""
    arr = arr if arr is not None else _zero_slab()
    for row in range(rows):
        t = t0 + row * row_step
        for w in warps:
            arr[0][w][row][0] = _tag(tag, t + 100)
            arr[0][w][row][1] = _tag(tag, t + 400)
    return arr


def _two_launch_slab(first_tag=0xFFFE, last_tag=0xFFFF):
    """An earlier launch reached rows 0..12; the last one (3 s later) rows 0..9."""
    arr = _tagged_wait_slab(first_tag, 10_000, rows=13)
    return _tagged_wait_slab(last_tag, 5_000_000_000, rows=10, arr=arr)


def test_the_stamp_carries_its_launch_in_the_high_bits():
    stamped, _ = stamp_source(TOY)
    assert "%clock64" in stamped and "%gridid" in stamped
    assert "and.b64 ttl, ttl, 0xFFFFFFFFFFFF" in stamped and "shl.b64 tgl, tgl, 48" in stamped
    from forgekernel.skills.timeline_probe import CLOCK_MASK, TAG_SHIFT
    assert TAG_SHIFT == 48 and CLOCK_MASK == 0xFFFFFFFFFFFF
    compile(stamped, "stamped.py", "exec")


def test_last_launch_keeps_only_the_last_launch_and_the_table_reads_one_launch():
    from forgekernel.skills.timeline_probe import last_launch
    mixed = _two_launch_slab()
    # read as it was before the tag: rows 10..12 of the earlier launch sit under the
    # last one — a restart, and a slot span across the host gap
    raw = [[[[int(v) & 0xFFFFFFFFFFFF for v in row] for row in w] for w in s] for s in mixed]
    assert coverage_span(reduce_slab(raw, WAIT_SITES)) > 4_000_000_000
    clocks, launches, stale, kernels = last_launch(mixed, WAIT_SITES)
    assert launches == 2 and kernels == 1
    assert stale == 3 * 2 * 2                            # rows 10..12, two warps, two sites
    assert clocks[0][4][0][0] == 5_000_000_100 and clocks[0][4][12][0] == 0
    res = reduce_slab(clocks, WAIT_SITES)
    assert res["restarts"] == 0 and coverage_span(res) == 9 * 1000 + 300
    assert res["roles"]["state"]["period"] == 1000
    # one launch in the slab: nothing dropped; an empty slab: nothing at all
    assert last_launch(_tagged_wait_slab(3, 10_000))[1:] == (1, 0, 1)
    assert last_launch(_zero_slab())[1:] == (0, 0, 0)


# One forward may launch SEVERAL stamped kernels — a split-KV kernel and its
# combine, GDN bwd's k2/k3/k4 — each its own launch (%gridid) of the same call.
# Keeping only the launch that wrote the latest clock dropped every kernel but
# the last one of the forward (a main kernel's 4104 stamps gone, the table left
# with the combine alone, span 0.03 of the call, still "usable"). The last launch
# is picked per kernel: sites tied by a shared role or by one launch writing both.
TWO_KERNEL_SITES = WAIT_SITES + ["comb.R.wait.begin", "comb.R.wait.end"]


def _two_kernel_slab():
    """Two calls of a forward: the main kernel (warps 4, 5) then its combine
    (warp 0, row 0). The earlier call's main reached rows 0..12, the last call's
    rows 0..9; the last call's combine overwrote every cell the earlier one wrote."""
    arr = _tagged_wait_slab(10, 10_000, rows=13)
    arr[0][0][0][2], arr[0][0][0][3] = _tag(11, 30_000), _tag(11, 30_500)
    arr = _tagged_wait_slab(20, 5_000_000_000, rows=10, arr=arr)
    arr[0][0][0][2], arr[0][0][0][3] = _tag(21, 5_000_020_000), _tag(21, 5_000_020_500)
    return arr


def test_each_kernel_of_the_forward_keeps_its_own_last_launch():
    from forgekernel.skills.timeline_probe import last_launch
    clocks, launches, stale, kernels = last_launch(_two_kernel_slab(), TWO_KERNEL_SITES)
    assert launches == 3 and kernels == 2
    assert stale == 3 * 2 * 2                            # the earlier main's rows 10..12 only
    assert clocks[0][4][0][0] == 5_000_000_100 and clocks[0][4][12][0] == 0
    assert clocks[0][0][0][2] == 5_000_020_000 and clocks[0][0][0][3] == 5_000_020_500
    res = reduce_slab(clocks, TWO_KERNEL_SITES)
    assert set(res["roles"]) == {"state", "comb"} and res["restarts"] == 0
    assert res["roles"]["state"]["period"] == 1000
    # the slot spans the call: main's first stamp to the combine's last
    assert coverage_span(res) == 20_500 - 100


COND_SITES = WAIT_SITES + ["state.R.rescale"]


def _conditional_site_slab():
    """One kernel, two launches over the same rows: only the earlier launch took
    the rescale path (site 2) — its surviving stamps sit on that site alone."""
    arr = _tagged_wait_slab(7, 10_000)
    for row in range(10):
        arr[0][4][row][2] = _tag(7, 10_000 + row * 1000 + 250)
    return _tagged_wait_slab(8, 5_000_000_000, arr=arr)


def test_a_site_only_an_earlier_launch_took_is_still_that_kernels_stale_stamp():
    from forgekernel.skills.timeline_probe import last_launch
    clocks, launches, stale, kernels = last_launch(_conditional_site_slab(), COND_SITES)
    assert (launches, stale, kernels) == (2, 10, 1)      # tied to the kernel by its role
    assert all(clocks[0][4][row][2] == 0 for row in range(10))


def test_last_launch_numpy_path_reads_the_same_as_the_nested_one():
    import pytest
    np = pytest.importorskip("numpy")
    if not hasattr(np, "asarray"):
        pytest.skip("numpy stand-in")
    from forgekernel.skills.timeline_probe import last_launch
    mixed = _two_launch_slab()
    nested, *n_counts = last_launch(mixed)
    arr, *a_counts = last_launch(np.array(mixed, dtype=np.int64))
    assert arr.dtype == np.int64 and arr.tolist() == nested and a_counts == n_counts
    for slab, sites in ((_two_kernel_slab(), TWO_KERNEL_SITES),
                        (_conditional_site_slab(), COND_SITES)):
        nested, *n_counts = last_launch(slab, sites)
        arr, *a_counts = last_launch(np.array(slab, dtype=np.int64), sites)
        assert arr.tolist() == nested and a_counts == n_counts


def _multi_workload_reading(device=(0.0093, 0.0101, 0.0200)):
    names = ("campaign", "local", "sharp")
    geo = (device[0] * device[1] * device[2]) ** (1 / 3)
    return {"correctness": {"pass": True, "max_diff": 0.0},
            "performance": {"device_median_ms": round(geo, 6), "aggregate": "geomean",
                            "n_workloads": 3,
                            "workloads": [{"name": n, "device_median_ms": d}
                                          for n, d in zip(names, device)]}}


def _mock_stamped_run(tp, monkeypatch, *, bench_slab, launcher_slab=None, launcher_rc=0):
    """The stamped bench leaves ``bench_slab`` (clock sampled at 1000 MHz); the
    campaign launcher leaves ``launcher_slab``. Returns the kernel texts each saw."""
    import os
    import sys
    fake = _FakeNumpy()
    monkeypatch.setitem(sys.modules, "numpy", fake)
    seen = {}
    monkeypatch.setattr(tp, "run_bench", lambda label, env=None: _multi_workload_reading())

    def stamped_bench(label, env=None, timeout=2400):
        with open(os.path.join(tp.ROOT, "kernel.py"), encoding="utf-8") as fh:
            seen["bench"] = fh.read()
        fake.save(env["TL_SLAB_OUT"], bench_slab)
        return _multi_workload_reading(), [1000.0, 1000.0], 0

    def launcher(env=None, timeout=900):
        with open(os.path.join(tp.ROOT, "kernel.py"), encoding="utf-8") as fh:
            seen["launcher"] = fh.read()
        if launcher_slab is not None:
            fake.save(env["TL_SLAB_OUT"], launcher_slab)
        return launcher_rc
    monkeypatch.setattr(tp, "run_bench_sampling_clock", stamped_bench)
    monkeypatch.setattr(tp, "run_slab_launcher", launcher)
    return seen


def test_workload_flag_scopes_the_bench_and_the_launcher_to_one_workload(tmp_path, monkeypatch, capsys):
    """The driver reads a group card's other meta workloads one job each
    (`--workload <name>`): the benches and the campaign launcher (whose ncu_window
    follows the same variable) read that one workload, so the slab and the span
    denominator agree; without the flag, the representative as before."""
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("# the card's campaign launch target\n")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    _mock_stamped_run(tp, monkeypatch, bench_slab=_two_launch_slab(),
                      launcher_slab=_two_launch_slab(first_tag=41, last_tag=42))
    inner, envs = tp.run_slab_launcher, []
    monkeypatch.setattr(tp, "run_slab_launcher", lambda env=None, timeout=900: (
        envs.append(dict(env or {})), inner(env=env, timeout=timeout))[1])
    assert tp.main(["--mode", "waits", "--workload", "wl7"]) == 0
    assert envs[-1][tp.WORKLOAD_SCOPE_ENV] == "wl7" and "TL_SLAB_OUT" in envs[-1]
    assert tp._scoped({"X": "1"}) == {tp.WORKLOAD_SCOPE_ENV: "wl7", "X": "1"}
    assert "workload scope: wl7" in capsys.readouterr().out
    assert tp.main(["--mode", "waits"]) == 0
    assert envs[-1][tp.WORKLOAD_SCOPE_ENV] == "representative"
    capsys.readouterr()
    assert tp.main(["--workload", "a b"]) == 0
    assert "TIMELINE-FAIL reason=bad-args" in capsys.readouterr().out


def test_the_table_is_the_campaign_launchers_last_launch_held_to_its_device_time(tmp_path, monkeypatch, capsys):
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("# the card's campaign launch target\n")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    seen = _mock_stamped_run(tp, monkeypatch, bench_slab=_two_launch_slab(),
                             launcher_slab=_two_launch_slab(first_tag=41, last_tag=42))
    assert tp.main(["--mode", "waits"]) == 0
    tl = parse_timeline(capsys.readouterr().out)
    assert tl["ok"] and tl["usable"] and tl["restarts"] == 0
    # 9300 cycles at 1000 MHz against the CAMPAIGN workload's 0.0093 ms — not the
    # 0.0126 ms geomean (0.74, short) and not the last workload's 0.02 ms
    assert tl["span_ratio"] == 1.0
    assert any(w.startswith("slab held 2 launches; read the last one from eval/ncu_run.py "
                            "(workload campaign), dropped 12 stale stamp(s)") for w in tl["warn"])
    # both processes ran a stamped kernel, each under its own nonce; the tree is restored
    assert "%gridid" in seen["bench"] and "%gridid" in seen["launcher"]
    assert seen["bench"] != seen["launcher"]
    assert (tmp_path / "kernel.py").read_text(encoding="utf-8") == TOY


def test_without_a_launch_target_the_benchs_last_launch_is_read_against_its_workload(tmp_path, monkeypatch, capsys):
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    seen = _mock_stamped_run(tp, monkeypatch, bench_slab=_two_launch_slab())
    assert tp.main(["--mode", "waits"]) == 0
    tl = parse_timeline(capsys.readouterr().out)
    assert "launcher" not in seen                       # no eval/ncu_run.py: never run
    assert tl["restarts"] == 0 and tl["span_ratio"] == round(9300 / 20000, 3)
    assert any("read the last one from the stamped bench (workload sharp)" in w for w in tl["warn"])


def test_a_launcher_that_leaves_no_stamps_falls_back_to_the_bench_by_name(tmp_path, monkeypatch, capsys):
    from forgekernel.skills import timeline_probe as tp
    (tmp_path / "kernel.py").write_text(TOY, encoding="utf-8")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("# the card's campaign launch target\n")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    _mock_stamped_run(tp, monkeypatch, bench_slab=_two_launch_slab(), launcher_rc=1)
    assert tp.main(["--mode", "waits"]) == 0
    tl = parse_timeline(capsys.readouterr().out)
    assert tl["restarts"] == 0
    assert any(w.startswith("eval/ncu_run.py left no stamps (rc=1)") for w in tl["warn"])
    assert (tmp_path / "kernel.py").read_text(encoding="utf-8") == TOY


TOY_TWO_KERNELS = TOY + """

@cute.kernel
def comb(x):
    _seg("comb.R.wait.begin")
    wait()
    _seg("comb.R.wait.end")
"""


def test_a_two_kernel_forward_reads_both_kernels_from_the_campaign_launch(tmp_path, monkeypatch, capsys):
    from forgekernel.skills import timeline_probe as tp
    _, sites = stamp_source(TOY_TWO_KERNELS, "waits")
    assert sites == TWO_KERNEL_SITES
    (tmp_path / "kernel.py").write_text(TOY_TWO_KERNELS, encoding="utf-8")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("# the card's campaign launch target\n")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    _mock_stamped_run(tp, monkeypatch, bench_slab=_two_kernel_slab(),
                      launcher_slab=_two_kernel_slab())
    assert tp.main(["--mode", "waits"]) == 0
    tl = parse_timeline(capsys.readouterr().out)
    assert set(tl["roles"]) == {"state", "comb"} and tl["restarts"] == 0
    assert any(w.startswith("slab held 3 launches; read the last launch of each of its 2 kernels "
                            "from eval/ncu_run.py (workload campaign), dropped 12 stale stamp(s)")
               for w in tl["warn"])
    assert (tmp_path / "kernel.py").read_text(encoding="utf-8") == TOY_TWO_KERNELS
