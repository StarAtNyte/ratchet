"""A lever price is a reading on ONE tree.

The pricing round banks its table under the best sha it ran on; the teacher
writes those numbers into the milestone as `lever:<name>` citations; the
student walks the cited rung several trees later. These tests pin the three
places the driver now compares the two trees:

* every priced row carries `priced_on` / `job`, and the recipes are banked
  with the table so a later tree can re-price a cited lever without a design
  turn;
* a KEEP that takes the watermark re-prices the milestone's cited levers whose
  price came from another tree (one job, base / +each), banks the readings,
  marks each citation in the milestone with its verdict, suspends a reversed /
  faded rung for the student, and hands the revisions to the teacher's next
  entry (which retires them once it has a fresh table);
* a reopen priced by a lever read on another tree is refused as stale.
"""
import json

from forgekernel.loop.driver import (
    _LEVER_PRICING_CMD,
    _cited_levers,
    _mark_lever_citations,
    _price_revision_note,
    _reprice_verdict,
    _student_prompt,
    _verify_reopen_price,
    run_inner_iteration,
    run_meta_step,
)
from test_loop_driver import (
    _DIAG,
    _MILESTONE,
    _PRICE_LOG_OK,
    _STALL_LOG_OK,
    EVAL,
    NCU,
    REC,
    FakeBackend,
    _ctx,
    _lever_ctx,
    _lever_design,
    _patching,
)

KF = "flash_attn_dsl/flash_bwd.py"
LEVER = "regsplit_184_120"
# a milestone rung citing a lever price the way a gdn teacher did
CITE = ("\n\n#### D0.15 aux\n- Action: r3 = split changed to 184/120, price +25.59% via lever:"
        + LEVER + " [MEASURED, job 885353]; see also lever:ring_lane (subset)\n")
# the re-price job: the same lever alone on the new tree, sign flipped
REPRICE_LOG = (
    "PRICE variants: base +regsplit_184_120\n"
    "LEVER regsplit_184_120 add=-11.63% remove=n/a add_ms=0.4896 remove_ms=None class=harmful "
    "spin_add=2640 spin_all_minus=None cubin_same_as_base=False\n"
    "STACK all=n/a sum_singles=-11.63% coupling=n/a all_pos=n/a all_pos_levers=none dropped=none\n"
    "PRICE noise_floor=1.14% metric=device_ms\n"
    "PRICE-OK levers=1 rounds=3 base_ms=0.5466 all_ms=None spin_base=177 spin_all=None "
    "tensor_base=9.9 tensor_all=None\n")


def _kernel(tmp_path, text="a\nc\ne\n"):
    kf = tmp_path / KF
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text(text)
    return kf


def _bank(ctx, best_ref, job, rows, spec):
    ev = ctx.evidence_dir
    ev.mkdir(parents=True, exist_ok=True)
    with (ev / "lever_prices.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": 0, "best_ref": best_ref, "job": job, "table": rows,
                             "spec_levers": spec}) + "\n")


def _priced_ctx(tmp_path, be, milestone=_MILESTONE + CITE):
    (tmp_path / "m.md").write_text(milestone, encoding="utf-8")
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.lever_pricing_rounds = 3
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx


_OLD_ROWS = [{"kind": "lever", "name": LEVER, "add_pct": 25.59, "remove_pct": None,
              "cls": "independent", "decisions": ["D0.4"]},
             {"kind": "stack", "all_pct": None, "noise_pct": 3.99}]
_OLD_SPEC = [{"name": LEVER, "edits": [{"old": "a", "new": "b"}]}]


def _revisions(ctx):
    p = ctx.evidence_dir / "price_revisions.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if p.is_file() else []


# --- citations & verdicts (pure) ------------------------------------------------

def test_cited_levers_reads_every_spelling_once():
    from forgekernel.loop.driver import _cited_price_jobs
    text = ("price +25.59% via lever:regsplit_184_120 [MEASURED, job 885353]\n"
            "reopen_price: +8% via lever：ring（independent）\n"
            "`lever:ring_lane` again, and lever:regsplit_184_120. Then a clever:trap\n"
            "lever:v1.2_ring stays whole, job 4242 elsewhere on the line")
    assert _cited_levers(text) == ["regsplit_184_120", "ring", "ring_lane", "v1.2_ring"]
    assert _cited_levers("") == []
    # the job a citation names is the reading it cites
    assert _cited_price_jobs(text, "regsplit_184_120") == ["885353"]
    assert _cited_price_jobs(text, "ring") == [] and _cited_price_jobs(text, "v1.2_ring") == ["4242"]


def test_reprice_verdicts_read_the_add_leg_against_the_new_noise_floor():
    old = {"kind": "lever", "add_pct": 25.59, "remove_pct": None}
    assert _reprice_verdict(old, {"kind": "lever", "add_pct": -11.63}, 1.14)[0] == "reversed"
    assert _reprice_verdict(old, {"kind": "lever", "add_pct": 0.4}, 1.14)[0] == "faded"
    assert _reprice_verdict(old, {"kind": "lever", "add_pct": 24.1}, 1.14)[0] == "confirmed"
    assert _reprice_verdict(old, {"kind": "lever", "add_pct": 12.0}, 1.14)[0] == "drifted"
    assert _reprice_verdict({"kind": "lever", "add_pct": 0.2}, {"kind": "lever", "add_pct": 9.0}, 1.0)[0] == "emerged"
    # no add on a table: the negated remove leg is what having the lever is worth
    assert _reprice_verdict({"kind": "lever", "add_pct": None, "remove_pct": -8.0},
                            {"kind": "lever", "add_pct": None, "remove_pct": 6.0}, 1.0)[0] == "reversed"
    assert _reprice_verdict(old, {"kind": "lever", "add_pct": None, "remove_pct": -24.0}, 1.0)[0] == "confirmed"
    assert _reprice_verdict(old, {"kind": "lever", "add_pct": None, "remove_pct": 11.63}, 1.0)[0] == "reversed"
    assert _reprice_verdict({"kind": "lever", "add_pct": None, "remove_pct": None},
                            {"kind": "lever", "add_pct": 3.0}, 1.0)[0] == "unmeasured"
    # a subset row prices as its delta
    assert _reprice_verdict({"kind": "variant", "delta_pct": 12.1},
                            {"kind": "variant", "delta_pct": -4.0}, 1.0)[0] == "reversed"


def test_marking_a_citation_replaces_its_earlier_marker():
    text = "price +25.59% via lever:x_1 [MEASURED, job 1]; and lever:x_1 again; lever:x_12 stays"
    once = _mark_lever_citations(text, "x_1", "[RE-PRICED @aaaa: add -1.00% — REVERSED]")
    assert once.count("[RE-PRICED @aaaa") == 2 and "lever:x_12 stays" in once
    assert "via lever:x_1 [MEASURED, job 1] [RE-PRICED @aaaa" in once
    twice = _mark_lever_citations(once, "x_1", "[RE-PRICED @bbbb: add +2.00% — CONFIRMED]")
    assert "[RE-PRICED @aaaa" not in twice and twice.count("[RE-PRICED @bbbb") == 2


# --- the inner-loop re-price ------------------------------------------------------

def test_new_best_reprices_a_stale_cited_lever_marks_the_milestone_and_suspends_the_rung(tmp_path):
    be = FakeBackend([REC("D2.3", 0.5466)], [EVAL(0.5466), NCU, REPRICE_LOG])
    ctx = _priced_ctx(tmp_path, be)
    _kernel(tmp_path)
    _bank(ctx, "sha_first_green", "885353", _OLD_ROWS, _OLD_SPEC)
    rec = run_inner_iteration(ctx, 8, metric_before=0.8947)
    assert rec.verdict == "KEEP"
    # one re-price job on the KEEP's own tree, then the workspace is put back
    gpu = [c for c in be.calls if c[0] == "gpu"]
    assert gpu == [("gpu", "all"), ("gpu", "ncu"), ("gpu", _LEVER_PRICING_CMD)]
    assert ("push", "fs-demo-reprice") in be.calls
    # checked out at the KEEP's tree before the recipe is applied, put back after the job
    resets = [i for i, c in enumerate(be.calls) if c == ("reset", rec.sha)]
    job_at = be.calls.index(("gpu", _LEVER_PRICING_CMD))
    assert resets[0] < job_at < resets[-1]
    spec = json.loads((tmp_path / "notes" / "levers.json").read_text())
    assert [lv["name"] for lv in spec["levers"]] == [LEVER]
    assert spec["levers"][0]["in_stack"] is False and spec["variants"] == [] and spec["roles"] is False
    # the revision: reversed, both trees named
    revs = _revisions(ctx)
    assert [(r["lever"], r["verdict"]) for r in revs] == [(LEVER, "reversed")]
    assert revs[0]["priced_on_before"] == "sha_first_green" and revs[0]["priced_on_now"] == rec.sha
    assert revs[0]["before"]["add_pct"] == 25.59 and revs[0]["now"]["add_pct"] == -11.63
    # the citation is marked in the milestone, after its [MEASURED] tag, and suspended
    text = ctx.milestone.read_text(encoding="utf-8")
    assert (f"via lever:{LEVER} [MEASURED, job 885353] [RE-PRICED @{rec.sha[:10]}" in text)
    assert "REVERSED; SUSPENDED by driver" in text
    # the subset citation the milestone never priced is untouched
    assert "lever:ring_lane (subset)" in text and text.count("[RE-PRICED") == 1
    # the new reading is banked under the new tree, recipe included
    banked = [json.loads(ln) for ln in (ctx.evidence_dir / "lever_prices.jsonl").read_text().splitlines()]
    assert banked[-1]["best_ref"] == rec.sha and banked[-1]["reprice"] is True
    row = next(lv for lv in banked[-1]["table"] if lv.get("kind") == "lever")
    assert (row["name"], row["add_pct"], row["priced_on"], row["decisions"]) == (LEVER, -11.63, rec.sha, ["D0.4"])
    assert banked[-1]["spec_levers"][0]["edits"] == [{"old": "a", "new": "b"}]
    # a price-check's partial table is history, never the teacher's table
    from forgekernel.loop.driver import _banked_lever_table
    assert _banked_lever_table(ctx, rec.sha) == []
    assert [lv["name"] for lv in _banked_lever_table(ctx, "sha_first_green") if lv["kind"] == "lever"] == [LEVER]
    # the student is told, on every turn, until the teacher's next entry
    prompt = _student_prompt(ctx, 9)
    assert "## Price-tag review" in prompt and f"`{LEVER}`" in prompt and "REVERSED" in prompt
    assert "do not land them this round" in prompt and "sha_first_" in prompt and rec.sha[:10] in prompt


def test_no_reprice_when_the_price_is_fresh_uncited_or_the_budget_is_short(tmp_path):
    # fresh: the banked table was read on the tree this KEEP produces (FakeBackend's first commit)
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _priced_ctx(tmp_path, be)
    _kernel(tmp_path)
    _bank(ctx, "sha1", "1", _OLD_ROWS, _OLD_SPEC)
    rec = run_inner_iteration(ctx, 1, metric_before=0.9)
    assert rec.sha == "sha1" and [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    assert _revisions(ctx) == [] and _price_revision_note(ctx) == ""
    # uncited: a stale table whose lever the milestone never cites
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _priced_ctx(tmp_path, be, milestone=_MILESTONE)
    _bank(ctx, "sha_old", "1", _OLD_ROWS, _OLD_SPEC)
    run_inner_iteration(ctx, 1, metric_before=0.9)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    # not a new best: nothing changed under the prices
    be = FakeBackend([REC("D2.3", 0.95)], [EVAL(0.95), NCU])
    ctx = _priced_ctx(tmp_path, be)
    _bank(ctx, "sha_old", "1", _OLD_ROWS, _OLD_SPEC)
    run_inner_iteration(ctx, 1, metric_before=0.9)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    # budget: < 60 min left — logged, no job, no revision
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _priced_ctx(tmp_path, be)
    ctx.deadline_epoch = 1000.0
    _bank(ctx, "sha_old", "1", _OLD_ROWS, _OLD_SPEC)
    run_inner_iteration(ctx, 1, metric_before=0.9)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    assert _revisions(ctx) == []
    # switched off per run
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _priced_ctx(tmp_path, be)
    ctx.reprice_on_new_best = False
    _bank(ctx, "sha_old", "1", _OLD_ROWS, _OLD_SPEC)
    run_inner_iteration(ctx, 1, metric_before=0.9)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]


def test_a_cited_lever_whose_edit_no_longer_applies_is_unverifiable_without_a_job(tmp_path):
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _priced_ctx(tmp_path, be)
    _kernel(tmp_path, "x\nc\ne\n")          # the recipe's `old` ("a") is gone from this tree
    _bank(ctx, "sha_old", "885353", _OLD_ROWS, _OLD_SPEC)
    rec = run_inner_iteration(ctx, 3, metric_before=0.9)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    revs = _revisions(ctx)
    assert [(r["lever"], r["verdict"]) for r in revs] == [(LEVER, "unverifiable")]
    assert "no longer applies" in revs[0]["note"] and revs[0]["now"] is None
    text = ctx.milestone.read_text(encoding="utf-8")
    assert f"[RE-PRICED @{rec.sha[:10]}: n/a — UNVERIFIABLE; its edit no longer applies" in text
    assert "SUSPENDED" not in text
    assert "as **unpriced**" in _student_prompt(ctx, 4)
    # a price banked before recipes were banked cannot be re-priced either
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _priced_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence-legacy"        # a fresh bank for this case
    _kernel(tmp_path)
    _bank(ctx, "sha_old", "885353", _OLD_ROWS, None)
    run_inner_iteration(ctx, 3, metric_before=0.9)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    revs = _revisions(ctx)
    assert [r["verdict"] for r in revs] == ["unverifiable"] and "before recipes were banked" in revs[0]["note"]


def test_a_failed_reprice_job_leaves_the_citation_unverified_and_the_workspace_restored(tmp_path):
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU, "PRICE-FAIL reason=lever-edit (boom)\n"])
    ctx = _priced_ctx(tmp_path, be)
    _kernel(tmp_path)
    _bank(ctx, "sha_old", "885353", _OLD_ROWS, _OLD_SPEC)
    rec = run_inner_iteration(ctx, 3, metric_before=0.9)
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls and ("reset", rec.sha) in be.calls
    revs = _revisions(ctx)
    assert [r["verdict"] for r in revs] == ["unverifiable"] and "refused the recipe" in revs[0]["note"]
    banked = [json.loads(ln) for ln in (ctx.evidence_dir / "lever_prices.jsonl").read_text().splitlines()]
    assert len(banked) == 1                                   # nothing banked for a failed job


# --- the teacher's entry -----------------------------------------------------------

def test_meta_entry_reads_the_revisions_and_retires_them_once_it_has_a_fresh_table(tmp_path):
    be = FakeBackend([_lever_design(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_OK])
    ctx = _lever_ctx(tmp_path, be)
    ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
    rv = {"ts": 0, "iter": 8, "lever": LEVER, "priced_on_before": "sha_first_green",
          "job_before": "885353", "before": {"add_pct": 25.59, "remove_pct": None},
          "priced_on_now": "sha_second", "job_now": "1788567310",
          "now": {"add_pct": -11.63, "remove_pct": None}, "verdict": "reversed",
          "note": "add +25.59% -> -11.63% (noise 1.14%): the sign flipped", "retired": False}
    (ctx.evidence_dir / "price_revisions.jsonl").write_text(json.dumps(rv) + "\n", encoding="utf-8")
    assert "## Price-tag review" in _student_prompt(ctx, 9)
    run_meta_step(ctx)
    diag = be.prompts[1]
    assert "## Price-tag review (measured by the driver in the inner loop" in diag and f"`{LEVER}`" in diag
    assert "REVERSED" in diag and "no SUSPENDED mark may be left in the milestone" in diag
    # a fresh table on the current best supersedes the revisions
    assert all(r["retired"] for r in _revisions(ctx))
    assert "## Price-tag review" not in _student_prompt(ctx, 10)
    # the fresh table carries provenance and its recipes, and says so to the teacher
    banked = [json.loads(ln) for ln in (ctx.evidence_dir / "lever_prices.jsonl").read_text().splitlines()]
    tab = banked[-1]
    assert tab["spec_levers"][0]["name"] == "ring" and tab["spec_levers"][0]["edits"] == [{"old": "a", "new": "b"}]
    assert all(lv["priced_on"] == tab["best_ref"] for lv in tab["table"])
    assert f"Measured tree = best `{tab['best_ref'][:10]}`" in diag and "The price tags hold only for this tree" in diag


def test_a_landed_lever_is_repriced_by_taking_it_out(tmp_path):
    """The student walked the rung: the tree carries the recipe's `new` text and
    not its `old`. The reversed recipe applies, its add is the cost of removing
    the lever, and the verdict reads that as the lever's present value."""
    undo_log = REPRICE_LOG.replace("add=-11.63%", "add=+11.63%")   # removing it makes the tree faster
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU, undo_log])
    ctx = _priced_ctx(tmp_path, be)
    _kernel(tmp_path, "b\nc\ne\n")
    _bank(ctx, "sha_old", "885353", _OLD_ROWS, _OLD_SPEC)
    rec = run_inner_iteration(ctx, 8, metric_before=0.9)
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls
    spec = json.loads((tmp_path / "notes" / "levers.json").read_text())
    assert spec["levers"][0]["edits"] == [{"old": "b", "new": "a"}]
    revs = _revisions(ctx)
    assert [(r["lever"], r["verdict"], r["landed"]) for r in revs] == [(LEVER, "reversed", True)]
    assert revs[0]["now"] == {"add_pct": None, "remove_pct": 11.63, "delta_pct": None, "cls": "harmful"}
    assert "taking it out" in revs[0]["note"] and "-remove -11.63%" in revs[0]["note"]
    banked = [json.loads(ln) for ln in (ctx.evidence_dir / "lever_prices.jsonl").read_text().splitlines()]
    row = next(lv for lv in banked[-1]["table"] if lv.get("kind") == "lever")
    assert row["landed"] is True and row["remove_pct"] == 11.63 and row["add_pct"] is None
    assert banked[-1]["spec_levers"][0]["edits"] == [{"old": "a", "new": "b"}]   # the forward recipe stays banked
    assert "remove +11.63%" in ctx.milestone.read_text(encoding="utf-8")
    assert rec.sha in ctx.milestone.read_text(encoding="utf-8")


def test_cited_price_jobs_ignore_the_drivers_reprice_marker():
    from forgekernel.loop.driver import _cited_price_jobs
    untagged = ("- r3: price +25.59% via lever:regsplit_184_120 [RE-PRICED @abcdef1234 job 886001: "
                "add -11.63% / remove n/a — REVERSED; SUSPENDED by driver]")
    assert _cited_price_jobs(untagged, "regsplit_184_120") == []
    tagged = ("- r3: price +25.59% via lever:regsplit_184_120 [MEASURED, job 885353] "
              "[RE-PRICED @abcdef1234 job 886001: add +20.00% — CONFIRMED]")
    assert _cited_price_jobs(tagged, "regsplit_184_120") == ["885353"]


def test_the_cited_reading_is_the_baseline_and_small_moves_are_not_repriced(tmp_path):
    """A verdict against the latest re-price would let a slow drift read
    confirmed at every step: the baseline is the reading the citation names
    (its job), else the oldest. And a best within the margin of the tree the
    lever was last read on is not re-read."""
    be = FakeBackend([REC("D2.3", 0.5)], [EVAL(0.5), NCU, REPRICE_LOG])
    ctx = _priced_ctx(tmp_path, be)
    _kernel(tmp_path)
    _bank(ctx, "sha_first", "885353", _OLD_ROWS, _OLD_SPEC)
    later = [{"kind": "lever", "name": LEVER, "add_pct": 20.0, "remove_pct": None, "cls": "independent"},
             {"kind": "stack", "all_pct": None, "noise_pct": 1.0, "base_ms": 0.9}]
    ev = ctx.evidence_dir
    with (ev / "lever_prices.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": 1, "best_ref": "sha_second", "job": "900", "table": later,
                             "spec_levers": _OLD_SPEC, "reprice": True}) + "\n")
    rec = run_inner_iteration(ctx, 8, metric_before=0.9)
    revs = _revisions(ctx)
    assert revs[0]["job_before"] == "885353" and revs[0]["before"]["add_pct"] == 25.59
    assert revs[0]["priced_on_before"] == "sha_first" and revs[0]["priced_on_now"] == rec.sha
    # margin: the latest reading's tree is within 3% of this best -> no job
    # (fresh workspaces: the seeded toolkits of the case above would make the
    # per-iteration census run here)
    w1 = tmp_path / "m1"
    w1.mkdir()
    be = FakeBackend([REC("D2.3", 0.89)], [EVAL(0.89), NCU])
    ctx = _priced_ctx(w1, be)
    _bank(ctx, "sha_old", "885353", _OLD_ROWS[:1] + [{"kind": "stack", "all_pct": None, "base_ms": 0.9}], _OLD_SPEC)
    run_inner_iteration(ctx, 9, metric_before=0.95)
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "all"), ("gpu", "ncu")]
    assert _revisions(ctx) == []
    # ...but a best 3% or more below it is
    w2 = tmp_path / "m2"
    w2.mkdir()
    be = FakeBackend([REC("D2.3", 0.87)], [EVAL(0.87), NCU, REPRICE_LOG])
    ctx = _priced_ctx(w2, be)
    _kernel(w2)
    _bank(ctx, "sha_old", "885353", _OLD_ROWS[:1] + [{"kind": "stack", "all_pct": None, "base_ms": 0.9}], _OLD_SPEC)
    run_inner_iteration(ctx, 9, metric_before=0.95)
    assert ("gpu", _LEVER_PRICING_CMD) in be.calls


def test_revisions_retire_at_a_teacher_entry_even_without_a_fresh_table(tmp_path):
    from test_loop_driver import _lever_ctx as lever_ctx
    be = FakeBackend(["LEVERS: none", _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = lever_ctx(tmp_path, be)
    ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
    rv = {"ts": 0, "iter": 8, "lever": LEVER, "priced_on_before": "sha_a", "job_before": "1",
          "before": {"add_pct": 25.59}, "priced_on_now": "sha_b", "job_now": "2",
          "now": {"add_pct": -11.63}, "verdict": "reversed", "note": "flipped", "retired": False}
    (ctx.evidence_dir / "price_revisions.jsonl").write_text(json.dumps(rv) + "\n", encoding="utf-8")
    run_meta_step(ctx)
    assert "## Price-tag review (measured by the driver in the inner loop" in be.prompts[1]
    assert all(r["retired"] for r in _revisions(ctx)) and "## Price-tag review" not in _student_prompt(ctx, 9)


def test_a_reopen_priced_by_a_lever_from_another_tree_is_stale(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    table = [{"kind": "lever", "name": "ring", "add_pct": 8.56, "remove_pct": -8.1,
              "cls": "independent", "priced_on": "shaA"},
             {"kind": "variant", "name": "ring_lane", "delta_pct": 12.4, "priced_on": "shaA"},
             {"kind": "stack", "all_pct": 8.56}]
    cite = "reopen_price: +8.6% via lever:ring"
    assert _verify_reopen_price(cite, [], table, ctx, [], best_ref="shaA") is None
    assert _verify_reopen_price(cite, [], table, ctx, []) is None            # tree unknown: no claim
    why = _verify_reopen_price(cite, [], table, ctx, [], best_ref="shaB")
    assert why and "priced on tree shaA" in why and "stale" in why and "levers.json" in why
    assert "stale" in _verify_reopen_price("+12% via lever:ring_lane", [], table, ctx, [], best_ref="shaB")
    # rows from before provenance was banked keep verifying
    legacy = [{"kind": "lever", "name": "ring", "add_pct": 8.56, "remove_pct": -8.1, "cls": "independent"}]
    assert _verify_reopen_price(cite, [], legacy, ctx, [], best_ref="shaB") is None
