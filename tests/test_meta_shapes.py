"""A group card's meta readings, one workload per job (loop/meta_shapes.py).

An audit of earlier group-card runs found the representative workload moved the same way as the rest
of the group only 61% of the times the rest moved by >= 2%, so lever prices read on it
alone misled the class column (and the phase closing that reads it). A card listing its
meta_workloads (the representative, the group's smallest, its largest) now prices each in
its own job and the driver merges the tables; stall / SASS pricing and the timeline probe
read the other workloads beside the representative's.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.loop import meta_shapes as ms
from forgekernel.loop.driver import (
    _LEVER_PRICING_CMD,
    _banked_timeline_of,
    _latest_timeline,
    _lever_price_block,
    _lever_price_submit,
    _lever_pricing_jobs_text,
    _meta_workload_readings,
    _price_log_verdict,
    _ref_workload_pricing,
    NoRunError,
    PricedTrees,
    parse_lever_prices,
    run_meta_step,
)
from test_loop_driver import (_DIAG, _ROLE_CORE_REF, _STALL_LOG_OK, FakeBackend, _lever_ctx,
                              _lever_design, _patching, _role_block)
from test_timeline_driver import TL_LOG, _tl_ctx


def _shape_log(wl, levers, *, noise=0.5, all_=6.0, ok=True, of=5, pos=None, same=None):
    """One workload's pricing job: ``levers`` = {name: (add, remove, class)}."""
    out = ["PRICE kernel_file=k.py variants: base " + " ".join(f"+{n}" for n in levers) + " all"]
    for name, (add, rem, cls) in levers.items():
        out.append(f"LEVER {name} add={add:+.2f}% remove={rem:+.2f}% add_ms=0.024 remove_ms=0.025 "
                   f"class={cls} spin_add=100 spin_all_minus=200 "
                   f"cubin_same_as_base={(same or {}).get(name, False)}")
    if pos:
        out.append(f"VARIANT all-pos levers={','.join(pos[1])} delta={pos[0]:+.2f}% ms=0.023 spin=90 "
                   f"q=auto: the stack without the harmful levers")
    out.append(f"STACK all={all_:+.2f}% sum_singles=+0.00% coupling=+0.00% "
               f"all_pos={'n/a' if not pos else f'{pos[0]:+.2f}%'} "
               f"all_pos_levers={','.join(pos[1]) if pos else 'none'} dropped={pos[2] if pos else 'none'}")
    out.append(f"PRICE-SCOPE workloads={wl} rep={wl} of={of}")
    out.append(f"PRICE noise_floor={noise:.2f}% metric=vs_ref_device sigma=0.30% rounds=2 outliers=none")
    out.append("PRICE-OK levers=2 rounds=2 base_ms=0.025 all_ms=0.024 spin_base=1000 spin_all=900 "
               "tensor_base=40.0 tensor_all=42.0" if ok else "PRICE-TIMED levers=2 rounds=2")
    return "\n".join(out) + "\n[gpu_entry.sh] task=cmd finished\n"


THREE = [
    ("wl0", _shape_log("wl0", {"ring": (4.0, -4.0, "independent"), "lane": (3.0, -1.0, "independent")},
                       noise=1.0), "101"),
    ("wl3", _shape_log("wl3", {"ring": (3.0, -3.0, "independent"), "lane": (-2.0, 0.1, "harmful")},
                       noise=1.0), "102"),
    ("wl7", _shape_log("wl7", {"ring": (5.0, -5.0, "independent"), "lane": (0.2, 0.0, "null")},
                       noise=1.0), "103"),
]


def test_merged_table_is_the_geomean_with_the_floor_of_a_mean_and_split_levers_flagged():
    log = ms.merge_lever_logs(THREE)
    assert _price_log_verdict(log) == "PRICE-OK"
    rows, stack, fail = parse_lever_prices(log)
    assert fail is None
    by = {r["name"]: r for r in rows}
    assert by["ring"]["add_pct"] == pytest.approx(100 * ((1.04 * 1.03 * 1.05) ** (1 / 3) - 1), abs=0.01)
    assert by["ring"]["cls"] == "independent"
    # +3% / -2% / +0.2%: up on one workload and down on another, each beyond its floor —
    # averaged it reads ~+0.4% (null); it is a size-dependent lever, never closed as null
    assert by["lane"]["cls"] == "shape_split"
    assert stack["noise_pct"] == pytest.approx(3 ** 0.5 / 3, abs=0.01)   # sqrt(sum f^2)/k
    assert stack["workload_scope"] == {"workloads": "wl0,wl3,wl7", "rep": "wl0", "of": "5"}
    assert stack["meta_shapes"]["jobs"] == ["101", "102", "103"] and stack["meta_shapes"]["missing"] == {}
    assert by["lane"]["shapes"]["wl3"] == {"add": -2.0, "remove": 0.1, "class": "harmful", "noise": 1.0,
                                           "spin_add": 100.0}
    assert set(stack["shapes"]) == {"wl0", "wl3", "wl7"}
    block = _lever_price_block([dict(r, kind=r.get("kind", "lever")) for r in rows] + [dict(stack, kind="stack")])
    assert "priced with one job on each of 3 workloads" in block and "`wl3`" in block
    assert "**shape_split**" in block and "This table's shape_split: `lane`" in block
    assert "only on the representative workload" not in block


def test_one_workload_still_reads_as_before():
    """A one-workload log carries none of the merged lines: parsed and rendered as ever."""
    rows, stack, _ = parse_lever_prices(THREE[0][1].replace("PRICE-SCOPE workloads=wl0 rep=wl0 of=5\n", ""))
    assert "shapes" not in rows[0] and "meta_shapes" not in stack
    assert ms.shape_block([dict(r, kind="lever") for r in rows] + [dict(stack, kind="stack")]) == ""
    rep_only = parse_lever_prices(THREE[0][1].replace("workloads=wl0 rep=wl0", "workloads=representative rep=wl0"))[1]
    assert "timed only on the representative workload `wl0`" in ms.scope_note(rep_only)


def test_noop_needs_identical_cubins_on_every_workload():
    same = {"lane": True}
    logs = [(wl, _shape_log(wl, {"ring": (4.0, -4.0, "independent"), "lane": (0.1, 0.0, "noop")},
                            same=same if wl != "wl7" else {}), str(i))
            for i, wl in enumerate(("wl0", "wl3", "wl7"))]
    by = {r["name"]: r for r in parse_lever_prices(ms.merge_lever_logs(logs))[0]}
    assert by["lane"]["cls"] == "null"           # one workload's cubin moved: not a noop
    logs[2] = ("wl7", _shape_log("wl7", {"ring": (4.0, -4.0, "independent"), "lane": (0.1, 0.0, "noop")},
                                 same=same), "2")
    by = {r["name"]: r for r in parse_lever_prices(ms.merge_lever_logs(logs))[0]}
    assert by["lane"]["cls"] == "noop"


def test_timing_copy_on_one_workload_and_a_missing_workload_and_per_workload_all_pos():
    lv = {"ring": (4.0, -4.0, "independent"), "lane": (-3.0, 0.0, "harmful")}
    logs = [("wl0", _shape_log("wl0", lv, pos=(5.0, ["ring"], "lane")), "1"),
            ("wl3", _shape_log("wl3", lv, ok=False, pos=(4.0, ["ring"], "lane")), "2")]
    log = ms.merge_lever_logs(logs, missing={"wl7": "no verdict"})
    # a role pass cut by the job clock on one workload: the timing copy, accepted as a table
    assert "PRICE-OK" not in log and _price_log_verdict(log) == "PRICE-TIMED"
    rows, stack, _ = parse_lever_prices(log)
    assert stack["all_pos_pct"] == pytest.approx(100 * ((1.05 * 1.04) ** 0.5 - 1), abs=0.01)
    assert stack["meta_shapes"]["missing"] == {"wl7": "no_verdict"}
    assert "`wl7` (no_verdict)" in ms.scope_note(stack)
    # a different harmful set per workload: no merged all-pos, the per-workload rows stay
    logs[1] = ("wl3", _shape_log("wl3", {"ring": (4.0, -4.0, "independent"), "lane": (0.2, -1.0, "coupled")},
                                 pos=None), "2")
    rows, stack, _ = parse_lever_prices(ms.merge_lever_logs(logs))
    assert stack["all_pos_pct"] is None and not [r for r in rows if r.get("kind") == "variant"]
    assert stack["shapes"]["wl0"]["all_pos_pct"] == 5.0


def _group_card(tree: Path, meta=("wl0", "wl3", "wl7")) -> None:
    p = tree / ms.PROBLEM_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"group": [{"name": f"wl{i}", "workload": {"axes": {"M": 8 * (i + 1)}}}
                                       for i in range(8)],
                             "meta_workloads": list(meta)}))


def test_card_meta_workloads_need_a_group_card_that_lists_them(tmp_path):
    assert ms.card_meta_workloads(tmp_path) == []
    _group_card(tmp_path)
    assert ms.card_meta_workloads(tmp_path) == ["wl0", "wl3", "wl7"]
    assert ms.card_workload_axes(tmp_path)["wl3"] == {"M": 32}
    _group_card(tmp_path, meta=("wl0",))
    assert ms.card_meta_workloads(tmp_path) == []
    assert ms.rounds_for(3, 3) == 2 and ms.rounds_for(3, 0) == 3 and ms.rounds_for(1, 3) == 1


def test_submit_prices_each_meta_workload_in_its_own_job(tmp_path):
    _group_card(tmp_path)
    be = FakeBackend([], [log for _, log, _ in THREE])
    ctx = _lever_ctx(tmp_path, be)
    log, job = _lever_price_submit(ctx, be, "b", _LEVER_PRICING_CMD, "meta/lever-price")
    # the role pass on the representative alone (the others' role columns were never read)
    assert [c[1] for c in be.calls if c[0] == "gpu"] == [
        f"{_LEVER_PRICING_CMD}{'' if wl == 'wl0' else ' --no-roles'} --workload {wl}"
        for wl in ("wl0", "wl3", "wl7")]
    assert "PRICE-SHAPES workloads=wl0,wl3,wl7" in log
    # another workload's job without a table: merged without it, and named
    be = FakeBackend([], [THREE[0][1], "nothing\n", THREE[2][1]])
    ctx = _lever_ctx(tmp_path, be)
    log, _ = _lever_price_submit(ctx, be, "b", _LEVER_PRICING_CMD, "x", rerun=False)
    assert parse_lever_prices(log)[1]["meta_shapes"]["missing"] == {"wl3": "no_verdict"}
    # the representative's own failure is the table's: its raw log, nothing else bought
    be = FakeBackend([], ["PRICE-FAIL reason=lever-edit (old not found)\n"])
    ctx = _lever_ctx(tmp_path, be)
    log, _ = _lever_price_submit(ctx, be, "b", _LEVER_PRICING_CMD, "x")
    assert log.startswith("PRICE-FAIL") and len([c for c in be.calls if c[0] == "gpu"]) == 1


def _priced(levers):
    return {wl: _shape_log(wl, {n: v[wl] for n, v in levers.items()}) for wl in ("wl0", "wl3", "wl7")}


def test_teacher_entry_prices_and_reads_every_meta_workload(tmp_path):
    """Plateau path on a group card: the representative's stall pricing as before, then
    the other two meta workloads' (one job each, rendered in their own section), then the
    lever table priced one job per workload and merged — split levers flagged."""
    _group_card(tmp_path)
    prices = _priced({"ring": {"wl0": (4.0, -4.0, "independent"), "wl3": (3.0, -3.0, "independent"),
                               "wl7": (5.0, -5.0, "independent")},
                      "lane": {"wl0": (3.0, -1.0, "independent"), "wl3": (-2.0, 0.1, "harmful"),
                               "wl7": (0.2, 0.0, "null")},
                      "nb_sym": {"wl0": (0.1, -3.0, "coupled"), "wl3": (0.0, -2.5, "coupled"),
                                 "wl7": (0.2, -2.8, "coupled")}})
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n", "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _STALL_LOG_OK, _STALL_LOG_OK.replace("900 45.00%", "700 35.00%"),
                      prices["wl0"], prices["wl3"], prices["wl7"]])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    gpu = [c[1] for c in be.calls if c[0] == "gpu"]
    assert gpu[1].endswith("--workload wl3") and gpu[2].endswith("--workload wl7")
    assert gpu[3:] == [f"{_LEVER_PRICING_CMD}{'' if wl == 'wl0' else ' --no-roles'} --workload {wl}"
                       for wl in ("wl0", "wl3", "wl7")]
    design, diag = be.prompts[0], be.prompts[1]
    # the design turn already reads the other workloads' readings
    assert "## Readings of the multi-workload tree's other workloads" in design and "### `wl3`" in design and "### `wl7`" in design
    assert "3 meta workloads" in design and "one GPU job for each" in design and "one GPU job." not in design
    assert "700 35.00%" in diag
    assert "priced with one job on each of 3 workloads" in diag and "This table's shape_split: `lane`" in diag
    banked = json.loads((tmp_path / "evidence" / "lever_prices.jsonl").read_text().splitlines()[-1])
    stack = next(r for r in banked["table"] if r["kind"] == "stack")
    assert stack["meta_shapes"]["workloads"] == ["wl0", "wl3", "wl7"]
    assert {r["name"]: r["cls"] for r in banked["table"] if r["kind"] == "lever"} == {
        "ring": "independent", "lane": "shape_split", "nb_sym": "coupled"}


def test_other_workloads_timelines_are_banked_beside_never_as_the_cards_own(tmp_path):
    _group_card(tmp_path)
    kf = tmp_path / "flash_attn_dsl" / "flash_bwd.py"
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text("def _seg(name):\n    pass\n")
    be = FakeBackend([], [_STALL_LOG_OK, TL_LOG, _STALL_LOG_OK, TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    ctx.stall_pricing = True
    block = _meta_workload_readings(ctx, PricedTrees(head_sha="sha9", best_sha="sha9"), "plateau")
    gpu = [c[1] for c in be.calls if c[0] == "gpu"]
    assert gpu[0].endswith("--workload wl3") and gpu[1].endswith("timeline_probe.py --workload wl3")
    assert gpu[3].endswith("timeline_probe.py --workload wl7")
    assert "### `wl3` (axes `{\"M\": 32}`)" in block and "Timeline (student)" in block
    # banked under the workload: never served as the card's own (representative) reading
    # (a backend without push_ref probes its HEAD: sha0)
    assert _latest_timeline(ctx) is None
    assert _banked_timeline_of(ctx, "sha0", "wl3") is not None and _banked_timeline_of(ctx, "sha0") is None


def test_same_sass_structure_is_said_once_not_listed_again():
    rows = [{"wl": "wl3", "same_as_rep": True, "ledger": "a\nb"},
            {"wl": "wl7", "diff_rep": "! rings: 2 vs 3"}]
    block = ms.workload_readings_block("wl0", rows, {"wl3": {"M": 32}}, "abc")
    assert "**identical axis by axis**" in block and "! rings: 2 vs 3" in block
    assert ms.workload_readings_block("wl0", [], {}, "abc") == ""


def test_a_missing_reason_with_separators_survives_the_price_shapes_line():
    """A reason is free text (a BackendError's message): a ';' in it used to cut the
    missing list where it stood, and the rest of the reason read as a workload."""
    why = "job failed (rc=1; node 42: gone at 5% left)"
    log = ms.merge_lever_logs(THREE[:2], missing={"wl7": why, "wl9": "budget"})
    meta = parse_lever_prices(log)[1]["meta_shapes"]
    assert meta["missing"] == {"wl7": "job_failed_(rc=1;_node_42:_gone_at_5%_left)", "wl9": "budget"}
    assert ms._dec_reasons(ms._enc_reasons({"a": "x%3By"})) == {"a": "x%3By"}
    assert ms._enc_reasons({}) == "none" and ms._dec_reasons("none") == {}


def test_texts_follow_the_number_of_workloads_actually_merged():
    # two of three merged: the counts say two, never "three"
    rows, stack, _ = parse_lever_prices(ms.merge_lever_logs(THREE[:2], missing={"wl7": "no verdict"}))
    table = [dict(r, kind=r.get("kind", "lever")) for r in rows] + [dict(stack, kind="stack")]
    note, block = ms.scope_note(stack), ms.shape_block(table)
    assert "priced with one job on each of 2 workloads" in note and "the geomean of these 2 workloads' readings" in note
    assert "`wl7` (no_verdict)" in note
    assert "the verdict over these 2 workloads together" in block and "three" not in block
    # only the representative left: not a merge, and no per-workload table under it
    rows, stack, _ = parse_lever_prices(ms.merge_lever_logs(THREE[:1], missing={"wl3": "budget",
                                                                               "wl7": "no verdict"}))
    table = [dict(r, kind=r.get("kind", "lever")) for r in rows] + [dict(stack, kind="stack")]
    note = ms.scope_note(stack)
    assert "one pricing job on each of the operator tree's 3 meta workloads" in note and "only the representative `wl0` got priced" in note
    assert "merged**" not in note and "the noise floor is the floor of that mean" not in note
    assert ms.shape_block(table) == ""
    assert "representative workload wl0 only" in ms.scope_log(stack)
    assert "merged over workloads wl0,wl3 " in ms.scope_log(
        parse_lever_prices(ms.merge_lever_logs(THREE[:2]))[1])


def test_the_design_turn_says_how_many_jobs_the_levers_buy(tmp_path):
    be = FakeBackend([], [])
    ctx = _lever_ctx(tmp_path, be)
    assert _lever_pricing_jobs_text(ctx) == "one GPU job"
    _group_card(tmp_path)
    text = _lever_pricing_jobs_text(ctx)
    assert "3 meta workloads (`wl0`, `wl3`, `wl7`" in text and "one GPU job for each" in text


def _with_ablation(log, *, delta=None, status="ok", verdict="ok"):
    """A shape log plus its ablation pass: ``verdict`` ok (ABLATE-OK), a fail reason
    (ABLATE-FAIL) or None (the job ended before the pass closed: no verdict)."""
    body = log.replace("[gpu_entry.sh] task=cmd finished\n", "")
    if verdict is not None:
        d = "n/a" if delta is None else f"{delta:+.2f}%"
        body += (f"ABLATE no_bar delta={d} ncu_ns=900 base_ncu_ns=1000 correctness=fail "
                 f"device_delta=n/a kernel=k status={status} h=the barrier is the exposed wait\n"
                 "PRICE ablation_noise_floor=1.00% metric=ncu_ns\n")
        body += ("ABLATE-OK ablations=1 rounds=2 base_ncu_ns=1000\n" if verdict == "ok"
                 else f"ABLATE-FAIL reason={verdict}\n")
    else:
        body += "ABLATE ROUND 0 base ncu_ns=1000\n"
    return body + "[gpu_entry.sh] task=cmd finished\n"


def test_a_workload_whose_ablation_did_not_close_is_named_never_read_as_flat():
    """Regression: another workload's ablation cut by the job clock (or failed) was
    dropped from the merge without a word, its cell a bare '—' the teacher could read as
    'no effect'. The merge now names it and why, and the table says it is unmeasured."""
    logs = [("wl0", _with_ablation(THREE[0][1], delta=10.0), "1"),
            ("wl3", _with_ablation(THREE[1][1], verdict=None), "2"),
            ("wl7", _with_ablation(THREE[2][1], verdict="base-hung (ncu timeout; rc=124)"), "3")]
    log = ms.merge_lever_logs(logs)
    assert "ABLATE-SHAPES missing=wl3:cut;wl7:fail:base-hung_(ncu_timeout%3B_rc=124)" in log
    rows, stack, _ = parse_lever_prices(log)
    abl = next(r for r in rows if r.get("kind") == "ablation")
    assert abl["delta_pct"] == pytest.approx(10.0)            # the workload that finished
    assert abl["shapes_missing"] == {"wl3": "cut", "wl7": "fail:base-hung_(ncu_timeout;_rc=124)"}
    table = [dict(r, kind=r.get("kind", "lever")) for r in rows] + [dict(stack, kind="stack")]
    block = ms.shape_block(table)
    row = next(ln for ln in block.splitlines() if ln.startswith("| ablation `no_bar`"))
    assert row.count("**incomplete**") == 2 and "—" not in row
    assert "the job ended before its ablation pass closed" in row and "ablation pass failed: base-hung_(ncu_timeout;_rc=124)" in row
    assert "**not** \"no change without it\"" in block
    # every workload's pass closed: no ABLATE-SHAPES line, no gap sentence
    full = ms.merge_lever_logs([(wl, _with_ablation(lg, delta=5.0), j) for wl, lg, j in THREE])
    assert "ABLATE-SHAPES" not in full
    rows, stack, _ = parse_lever_prices(full)
    table = [dict(r, kind=r.get("kind", "lever")) for r in rows] + [dict(stack, kind="stack")]
    assert "incomplete" not in ms.shape_block(table)


def test_an_ablation_the_pass_budget_never_reached_reads_unmeasured():
    logs = [("wl0", _with_ablation(THREE[0][1], delta=10.0), "1"),
            ("wl3", _with_ablation(THREE[1][1], status="budget"), "2")]
    rows, stack, _ = parse_lever_prices(ms.merge_lever_logs(logs))
    table = [dict(r, kind=r.get("kind", "lever")) for r in rows] + [dict(stack, kind="stack")]
    block = ms.shape_block(table)
    row = next(ln for ln in block.splitlines() if ln.startswith("| ablation `no_bar`"))
    assert "+10.00% ok" in row and "**incomplete** (the ablation budget ran out" in row


class _RefBackend(FakeBackend):
    """The reference tree's backend: an exception in ``gpu_logs`` is raised, and every
    submit moves the job id (a job that launched) unless it is a NoRunError."""
    last_job_id = ""

    def submit_gpu(self, branch, mode):
        self.calls.append(("gpu", mode))
        out = self.gpu_logs.pop(0)
        if not isinstance(out, NoRunError):
            self.last_job_id = str(len(self.calls))
        if isinstance(out, Exception):
            raise out
        return out


def _ref_wl_ctx(tmp_path, gpu_logs):
    _group_card(tmp_path)
    ctx = _lever_ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    ref = tmp_path / "_ref"
    (ref / "eval").mkdir(parents=True)
    (ref / "eval" / "ncu_run.py").write_text("pass\n")
    _group_card(ref)
    ctx.ref_price_dir, ctx.ref_price_backend = ref, _RefBackend([], gpu_logs)
    return ctx


_REF_OK = _STALL_LOG_OK + _role_block(_ROLE_CORE_REF)


def _ref_jobs(ctx):
    return [c for c in ctx.ref_price_backend.calls if c[0] == "gpu"]


def test_a_reference_workload_pricing_without_a_reading_is_retried_once_the_tree_changed(tmp_path):
    """Regression: ref_price_<wl>.log was banked whatever came back -- a ROLE-FAIL, a
    truncated log -- and never priced again. Only a reading is banked now; the rest is an
    attempt, retried by the rule the representative's reference pricing keeps."""
    ctx = _ref_wl_ctx(tmp_path, ["ROLE-FAIL reason=too-few-samples (120 < 500)\n", _REF_OK])
    ev = ctx.evidence_dir
    assert _ref_workload_pricing(ctx, "wl3", "plateau") is None
    assert not (ev / "ref_price_wl3.log").exists()
    head = (ev / "ref_price_wl3.attempt1.log").read_text().splitlines()[0]
    assert head.startswith("[ref-price] key=") and "verdict=ROLE-FAIL_reason=too-few-samples" in head
    assert _ref_workload_pricing(ctx, "wl3", "plateau") is None      # same tree: it would repeat
    assert len(_ref_jobs(ctx)) == 1
    (ctx.ref_price_dir / "eval" / "ncu_run.py").write_text("pass  # window fixed\n")
    assert _ref_workload_pricing(ctx, "wl3", "plateau") == _REF_OK
    assert (ev / "ref_price_wl3.log").read_text() == _REF_OK
    assert _ref_workload_pricing(ctx, "wl3", "audit") == _REF_OK      # banked: read back, no job
    assert len(_ref_jobs(ctx)) == 2 and _ref_jobs(ctx)[0][1].endswith("--workload wl3")


def test_a_reference_workload_log_with_no_verdict_is_retried_up_to_the_cap(tmp_path):
    ctx = _ref_wl_ctx(tmp_path, ["[gpu_entry.sh] task=cmd started\n"] * 4)
    for _ in range(5):
        assert _ref_workload_pricing(ctx, "wl7", "plateau") is None
    assert len(_ref_jobs(ctx)) == 3                                   # the archive lost it: retried
    assert "verdict=none" in (ctx.evidence_dir / "ref_price_wl7.attempt3.log").read_text()


def test_a_reference_workload_job_that_never_ran_is_no_attempt(tmp_path):
    ctx = _ref_wl_ctx(tmp_path, [NoRunError("GPU not up")] * 3 + [_REF_OK])
    for _ in range(4):
        assert _ref_workload_pricing(ctx, "wl3", "plateau") is None
    assert len(_ref_jobs(ctx)) == 3                                   # bounded per session
    assert not list(ctx.evidence_dir.glob("ref_price_wl3.attempt*.log"))
    ctx.ref_wl_price_submit_failures.clear()                          # a new session
    assert _ref_workload_pricing(ctx, "wl3", "plateau") == _REF_OK


def test_a_reference_workload_job_that_failed_after_launch_waits_for_a_changed_tree(tmp_path):
    from forgekernel.loop.driver import BackendError
    ctx = _ref_wl_ctx(tmp_path, [BackendError("poll timeout after 7200s")])
    assert _ref_workload_pricing(ctx, "wl3", "plateau") is None
    assert "verdict=job-failed" in (ctx.evidence_dir / "ref_price_wl3.attempt1.log").read_text()
    assert _ref_workload_pricing(ctx, "wl3", "plateau") is None
    assert len(_ref_jobs(ctx)) == 1


def test_a_banked_reference_workload_log_without_a_reading_is_priced_again(tmp_path):
    """A run resumed from before the rule: its banked ROLE-FAIL log is an attempt."""
    ctx = _ref_wl_ctx(tmp_path, [_REF_OK])
    ctx.evidence_dir.mkdir()
    (ctx.evidence_dir / "ref_price_wl3.log").write_text("ROLE-FAIL reason=no-report\n")
    assert _ref_workload_pricing(ctx, "wl3", "plateau") == _REF_OK
    assert "key=legacy" in (ctx.evidence_dir / "ref_price_wl3.attempt1.log").read_text()


_DIFF = """=== dossier diff (wl0 vs wl3) ===
  = loops: {'copy-producer': 4}
  ! mbar_objs:
      wl0: 1
      wl3: 2
  = rings: []
  = edges: []
ORCH-DIFF mismatches=1"""
_ROLES = """[parse] page carried 8 launches; using launch 6
[join] 66 sampled PCs onto 18 regions; 95.0% of 855 stall samples resolved
[tensor] active=8.7% source=ncu-aggregate duration_ns=32160.0 exposed_frac=0.913
ROLE label=copy-producer regions=4 samples=309 share=38.1 exposed_pct=34.7 exposed_ns=11170 stalls=long_scoreboard:84,short_scoreboard:6
ROLE label=spin regions=3 samples=423 share=52.1 exposed_pct=47.5 exposed_ns=15290 stalls=long_scoreboard:99
REGION r13 spin          [0x7fc7b55b2910..0x7fc7b55b2940]     4 instrs share=51.8 stalls=long_scoreboard:100
ROLE-JSON: {"duration_ns": 32160.0, "roles": {}}"""
_LEDGER = ("[ledger] keying=source-line (95% of samples line-resolved)\n\n"
           "--- ledger rows (cumulative to >= 90% of 855 true-stall samples) ---\n"
           + "".join(f"     {500 - i} 5.00% (cum  {i}.0%)  kernel.py:{300 + i}      BRA 0x1   | long_scoreboard:100%\n"
                     for i in range(10))
           + "\n--- opcode family x dominant stall ---\n     415 48.54%  NANOSLEEP  | long_scoreboard=100%\n")
# the driver's rendering (driver._timeline_tables), with its shared strings
_TL = ("Checks: mode=all, numeric=identical, perturbation 1.0166×, span/device 0.696.\n"
       f"{ms.TL_COVERAGE_NOTE} (partial)**: this table's stamps cover only 0.696 of device time.\n"
       f"{ms.TL_SELFCHECK_NOTE}**: row-counter restarts 8/400\n\n"
       "| role | warps | period per block | busy | wait | wrap (unmarked tail) | period IQR |\n|---|---|---|---|---|---|---|\n"
       "| tma | 8 | 614 | 0 | 541 | 77 | 0.875 |\n\n"
       f"**epi** {ms.TL_SEGMENT_LEDGER_HEADING}\n"
       "| segment | kind | median | p25 | p75 | n |\n|---|---|---|---|---|---|\n\n"
       f"**tma** {ms.TL_SEGMENT_LEDGER_HEADING}\n"
       "| segment | kind | median | p25 | p75 | n |\n|---|---|---|---|---|---|\n"
       "| `tma.a` → `tma.b` | wait | 541 | 323 | 864 | 160 |\n")


def test_other_workloads_section_keeps_what_can_differ_and_drops_what_repeats():
    """The section was 24.6k chars / 348 lines on the B200 e2e logs (two workloads,
    the reference on both), most of it what the representative's blocks already say."""
    rows = [{"wl": "wl3", "diff_rep": _DIFF, "diff_ref": _DIFF.replace("wl0", "ref").replace("wl3", "student"),
             "roles": _ROLES, "ref_roles": _ROLES.replace("share=38.1", "share=9.7"), "ledger": _LEDGER,
             "timeline": _TL}]
    block = ms.workload_readings_block("wl0", rows, {"wl3": {"M": 32}}, "abc",
                                       rep_roles=_ROLES.replace("share=52.1", "share=53.0"))
    # the dossier: only the axes that differ, with how many were the same
    assert "! mbar_objs:" in block and "wl3: 2" in block and "= loops" not in block
    assert "the other 3 axes are the same" in block
    # the role accounts side by side: student / reference / student on the representative
    assert "| copy-producer | 38.1% / 11170 ns · long_scoreboard:84 | 9.7% / 11170 ns" in block
    assert "| spin | 52.1% / 15290 ns · long_scoreboard:99 | 52.1% / 15290 ns · long_scoreboard:99 | 53.0%" in block
    assert "| tensor active | 8.7% | 8.7% | 8.7% |" in block
    assert "REGION r13" not in block and "ROLE-JSON" not in block and "[parse]" not in block
    # the ledger's head rows, not its opcode table
    assert "kernel.py:300" in block and "kernel.py:305" in block and "kernel.py:306" not in block
    assert "opcode family" not in block
    # the timeline's tables without the per-run caveats or an empty segment table
    assert "| `tma.a` → `tma.b` | wait | 541" in block and "**epi** segment ledger" not in block
    assert "Coverage note" not in block.split("(2 more")[0] and "(2 more coverage / probe self-check notes left out" in block
    # no reference reading on a workload: said in the table
    rows[0].pop("ref_roles")
    assert "(missing: this run has no reference pricing tree, or the reference was not priced on this workload)" in ms.workload_readings_block(
        "wl0", rows, {}, "abc")


def test_other_workloads_section_is_capped_and_says_where():
    rows = [{"wl": f"wl{i}", "diff_rep": _DIFF, "roles": _ROLES, "ledger": _LEDGER, "timeline": _TL}
            for i in range(1, 9)]
    full = ms.workload_readings_block("wl0", rows, {}, "abc", limit=10 ** 6)
    capped = ms.workload_readings_block("wl0", rows, {}, "abc", limit=3000)
    assert len(full) > 3000 >= len(capped)
    assert "section truncated: over the 3000-character cap" in capped and "must not be treated as read" in capped
    assert capped.count("```") % 2 == 0                      # no code fence left open
    assert len(ms.workload_readings_block("wl0", rows[:2], {}, "abc")) <= ms.READINGS_MAX_CHARS


def test_a_remove_leg_that_disagrees_alone_is_not_a_shape_split():
    """Only the ADD leg splits. In earlier runs 4 of the 8 shape_splits disagreed on the
    remove leg alone -- l2pf_presync slowed all three workloads when added (-2.28 / -1.45 /
    -7.21%) and still read shape_split, so its phase never closed. A remove leg that disagrees
    says how the lever couples with the stack at each size: the merged class decides."""
    shapes = [
        ("wl23", _shape_log("wl23", {"pre": (-2.28, -1.74, "coupled"), "pin": (-7.65, 0.1, "harmful")},
                            noise=1.16), "1"),
        ("wl36", _shape_log("wl36", {"pre": (-1.45, -2.01, "coupled"), "pin": (11.11, -5.1, "independent")},
                            noise=0.85), "2"),
        ("wl16", _shape_log("wl16", {"pre": (-7.21, 0.75, "harmful"), "pin": (0.1, 0.0, "null")},
                            noise=0.5), "3"),
    ]
    rows, _, fail = parse_lever_prices(ms.merge_lever_logs(shapes))
    by = {r["name"]: r for r in rows}
    assert fail is None
    assert by["pre"]["cls"] not in ("shape_split", "independent", "dependent")   # never read as a gain
    assert by["pin"]["cls"] == "shape_split"         # the add leg itself goes both ways
    assert ms.is_split([-2.28, -1.45, -7.21], [1.16, 0.85, 0.5]) is False
