"""Three review findings that bite a closed-source (blackbox) rerun:

1. the reference dossier was banked from the FIRST block in the recon log with
   no kernel identity — a chain replay prices several kernels;
2. a half-failed pricing pair (best + HEAD) handed the teacher the surviving
   face under the both-succeeded labels, and "student best" was printed when
   there was no best;
3. the diagnose prompt promised G5 = KB recipe + a student in-segment
   reordering, while the driver only ran the KB branch."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import (  # noqa: E402
    _DIAG, _MILESTONE, _STALL_LOG_OK, FakeBackend, _ctx, _patching, _seed_keep,
)

from forgekernel.loop.driver import (  # noqa: E402
    BackendError, PricedTrees, _orch_diff_views, _pick_ref_dossier,
    _price_student_trees, _save_ref_orch_dossier, _student_prompt, run_meta_step,
)
from forgekernel.inner.trajectory import load_records  # noqa: E402


def _block(instrs, summary):
    return ("=== ORCH DOSSIER BEGIN ===\n"
            f"[parse] {instrs} instructions, unparsed=3\n"
            "=== regions (4, 2 loops, 0 spin) ===\n"
            "ORCH-SUMMARY-JSON: " + json.dumps(summary, sort_keys=True) + "\n"
            "=== ORCH DOSSIER END ===\n")


_FAIL_SUMM = {"loops": {}, "mbar_objs": 0, "cache_ops": [], "verdict": "ORCH-FAIL no-sync-sites"}
_OK_SUMM = {"loops": {"copy-producer": 1, "mma-consumer": 1}, "mbar_objs": 3,
            "cache_ops": ["LDGSTS.E.BYPASS.LTC256B.128"], "verdict": "ORCH-OK"}
_ORCH_OK_LINE = ("ORCH-OK regions=4 loops=2 nbar_ids=1 mbar_objs=3 resolved=3 "
                 "edges=2 itc_candidates=0\n")

# A chain-replay pricing log: a bystander kernel's (FAIL) dossier first, then the
# kernel under judgment, named by the dominant-launch line the runner prints.
_CHAIN_LOG = (
    "[stall_price] dominant launch index=0 name=topk_kernel\n"
    + _block(310, _FAIL_SUMM) + "ORCH-FAIL no-sync-sites\n"
    "[stall_price] dominant launch index=2 name=sparse_attn_fwd_kernel\n"
    + _block(3520, _OK_SUMM) + _ORCH_OK_LINE
    + "STALL-OK kernel=sparse_attn_fwd_kernel samples=2000 rows=3 coverage=92.0\n")


# --- 1. reference dossier: pick the kernel, stamp its identity ----------------

def test_ref_dossier_picks_the_ok_largest_block_and_stamps_the_kernel():
    text, note = _pick_ref_dossier(_CHAIN_LOG)
    assert text.startswith("# banked: block 2/2 kernel=sparse_attn_fwd_kernel "
                           "instrs=3520 verdict=OK\n")
    assert "LDGSTS.E.BYPASS.LTC256B.128" in text and "no-sync-sites" not in text
    assert "ORCH DOSSIER BEGIN" not in text and "\nORCH-OK" not in text
    assert "not banked: block 1: topk_kernel instrs=310 ORCH-FAIL" in note


def test_ref_dossier_single_block_keeps_the_old_contract():
    log = _block(3520, _OK_SUMM) + _ORCH_OK_LINE
    text, note = _pick_ref_dossier(log)
    assert note == "" and text.startswith("# banked: block 1/1 kernel=? instrs=3520 verdict=OK\n")
    assert _pick_ref_dossier("no dossier here") is None
    # two OK blocks: the larger kernel wins, a same-size retry keeps the later one
    two = (_block(900, _OK_SUMM) + _ORCH_OK_LINE + _block(3520, _OK_SUMM) + _ORCH_OK_LINE)
    assert _pick_ref_dossier(two)[0].startswith("# banked: block 2/2")
    retry = _block(3520, _OK_SUMM) + _ORCH_OK_LINE + _block(3520, _OK_SUMM) + _ORCH_OK_LINE
    assert _pick_ref_dossier(retry)[0].startswith("# banked: block 2/2")


def test_save_ref_orch_dossier_banks_the_picked_block(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    _save_ref_orch_dossier(ctx, _CHAIN_LOG)
    banked = (tmp_path / "evidence" / "ref_orch_dossier.txt").read_text()
    assert banked.startswith("# banked: block 2/2 kernel=sparse_attn_fwd_kernel")
    from forgekernel.loop.driver import _orch_summary
    assert _orch_summary(banked)["cache_ops"] == ["LDGSTS.E.BYPASS.LTC256B.128"]


# --- 2. half-failed pricing pair: the missing face is named -------------------

class _SecondJobDies(FakeBackend):
    def submit_gpu(self, branch, mode):
        self.calls.append(("gpu", mode))
        if len([c for c in self.calls if c[0] == "gpu"]) == 2:
            raise BackendError("archive lost")
        return self.gpu_logs.pop(0)


def test_price_student_trees_names_the_dead_head_job(tmp_path):
    be = _SecondJobDies([], [_STALL_LOG_OK + _block(3520, _OK_SUMM) + _ORCH_OK_LINE])
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.evidence_dir = tmp_path / "evidence"
    _seed_keep(ctx, "D1.1", 2.5, sha="prevbest0001")   # best != HEAD ("sha0")
    priced = _price_student_trees(ctx, load_records(ctx.status_path))
    assert priced.ledger_tree == "best" and priced.ledger is not None
    assert priced.best_dossier is not None and priced.head_dossier is None
    assert len(priced.missing) == 1 and priced.missing[0].startswith("HEAD sha0: pricing job failed")
    view = _orch_diff_views(ctx, priced)
    assert "Faces absent this time" in view and "HEAD sha0: pricing job failed" in view
    assert "ref: recon left no comparable dossier summary" in view     # no ref dossier banked either
    assert "ref (recon time) vs HEAD" not in view               # no phantom face


def test_price_student_trees_reads_head_when_there_is_no_best(tmp_path):
    be = FakeBackend([], [_block(3520, _OK_SUMM) + _ORCH_OK_LINE])
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.evidence_dir = tmp_path / "evidence"
    priced = _price_student_trees(ctx, [])
    assert priced.ledger_tree == "HEAD" and priced.best_sha is None
    assert priced.head_dossier is not None
    # the ledger self-check did not pass in this log: named, not silent
    assert priced.missing == ["HEAD sha0: ledger self-check failed, no ledger (dossier present)"]


def test_diagnose_prompt_labels_the_priced_tree_honestly(tmp_path):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK])
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    run_meta_step(ctx)          # no KEEP on record: the ledger is HEAD's
    prompt = be.prompts[0]
    assert "student HEAD kernel" in prompt and "student best kernel" not in prompt


def test_orch_diff_views_silent_only_when_nothing_at_all(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert _orch_diff_views(ctx, PricedTrees()) is None
    view = _orch_diff_views(ctx, PricedTrees(missing=["best abc: pricing job failed"]))
    assert view is not None and "best abc: pricing job failed" in view


# --- 3. G5: the student half of the promise exists ----------------------------

def test_g5_meta_step_hands_the_reordering_directive_to_the_student(tmp_path):
    be = FakeBackend(
        ["cause_class: G5\ndeficient_locus: seg:HG-run-2\n"
         "patch_direction: interleave the two EW chains across the DEPBAR",
         "PROBE_REQUEST: notes/kb_probe.py", "RECIPE_CERTIFIED: hw_kb/cutedsl/sm90/ilp.md"],
        ["PROBE PASS max_diff=0.0"],
    )
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "kb-example" and res.recipe == "hw_kb/cutedsl/sm90/ilp.md"
    assert res.student_directive and "seg:HG-run-2" in res.student_directive
    assert "interleave the two EW chains" in res.student_directive
    assert "together with the KB recipe just certified" in res.student_directive
    ctx.fresh_recipe, ctx.g5_note = res.recipe, res.student_directive
    prompt = _student_prompt(ctx, 7)
    assert "G5 intra-segment ILP" in prompt and "seg:HG-run-2" in prompt and "hw_kb/cutedsl/sm90/ilp.md" in prompt


def test_g5_directive_survives_an_uncertified_recipe_and_g3_has_none(tmp_path):
    be = FakeBackend(
        ["cause_class: G5\ndeficient_locus: seg:EW-3\npatch_direction: hoist the MUFU",
         "PROBE_REQUEST: notes/kb_probe.py", "PROBE_REQUEST: notes/kb_probe.py"],
        ["PROBE FAIL max_diff=1.0", "PROBE FAIL max_diff=1.0"],
    )
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    res = run_meta_step(ctx)
    assert res.certified is False and res.recipe is None
    assert res.student_directive and "KB recipe could not be certified" in res.student_directive
    be = FakeBackend(
        ["cause_class: G3\ndeficient_locus: KB:swizzle\npatch_direction: certify flat-tv",
         "PROBE_REQUEST: notes/kb_probe.py", "RECIPE_CERTIFIED: hw_kb/cutedsl/sm90/swizzle.md"],
        ["PROBE PASS max_diff=0.0"],
    )
    assert run_meta_step(_ctx(tmp_path, be)).student_directive is None
