"""Out-of-run gap queue: candidates rows nobody in the run will verify.

VERIFY `none` rows and expressibility guesses without an [EXPR:] ladder
verdict are banked at recon completion, surfaced by the harvest turn as a
[pending certification] list, and never silently dropped (regression: a sparse cache-bits row)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import FakeBackend, _ctx  # noqa: E402

from forgekernel.loop.driver import (  # noqa: E402
    _persist_gap_queue, load_gap_queue, parse_gap_queue, run_kb_harvest,
)

_CANDIDATES = """
## Candidate A

| Decision | Value | VERIFY | Notes |
|---|---|---|---|
| mbarrier object count | 10 (1+4+4+1) | `ncu-derived` | countable, usable as a gate |
| **`DEPBAR.LE 0x1`** | 1 WGMMA left in flight when the KV buffer is released | `none` | **the most important `none`** |
| supply issue order | index `LDG` interleaved with `LDGSTS` | `sass-derived` | dossier supply axis |
| cache bits | `.BYPASS` + `.LTC256B`; `CREATEPOLICY = 0` | `sass-derived` | may not be expressible in the DSL |
| wait encoding | try_wait tight spin | `sass-derived` | may not be expressible in the DSL [EXPR: A=mbarrier_try_wait; B=-; C=-] |

### A.8 Numerics

| Item | Value | VERIFY |
|---|---|---|
| precision contract | fp32 accumulation | `correctness` |
| mask wait position | after chain0 | none |

## Candidate B (control)

| Decision | Value | VERIFY | Notes |
|---|---|---|---|
| **`DEPBAR.LE 0x1`** | 1 WGMMA left in flight when the KV buffer is released | `none` | duplicate row |
"""


def test_parse_gap_queue_collects_none_rows_and_unverified_guesses():
    rows = parse_gap_queue(_CANDIDATES)
    by = {r.decision: r for r in rows}
    assert set(by) == {"`DEPBAR.LE 0x1`", "cache bits", "mask wait position"}
    assert by["`DEPBAR.LE 0x1`"].reason == "verify-none"
    assert by["cache bits"].reason == "expressibility-guess"
    assert by["cache bits"].value == "`.BYPASS` + `.LTC256B`; `CREATEPOLICY = 0`"
    assert by["cache bits"].remark == "may not be expressible in the DSL"
    assert by["mask wait position"].reason == "verify-none" and by["mask wait position"].remark == ""
    # a guess WITH an [EXPR:] verdict is not a gap; a verified row is not a gap
    assert ("wait encoding" not in by and "supply issue order" not in by
            and "mbarrier object count" not in by)
    # the duplicate row in candidate B collapses
    assert len(rows) == 3
    assert "unverified expressibility guess" in by["cache bits"].line()
    assert parse_gap_queue("no tables here\n| a | b |\n") == []


# The spelling the recon prompt actually prescribes and most runs' tables carry:
# the channel is an inline `[VERIFY: x]` tag parked in a `tag` / `source` column,
# with no column named VERIFY at all (fp8, sparse, bwd-GQA, fwd-MHA runs — all
# of which parsed to zero rows before the tag was read).
_CANDIDATES_INLINE = """
## Candidate A

| field | value | tag |
|---|---|---|
| tile bM x bN x bK | **256 x 128 x 128** | `[VERIFY: ncu-derived]` — smem budget closes |
| MMA atom | **wgmma m64n128k32 e4m3** | `[VERIFY: none]` (SASS-only; no NCU counter) |
| K-loop schedule | wait full -> 4x wgmma -> `DEPBAR.LE 0` | `[VERIFY: none]` (SASS); its consequence is `[VERIFY: ncu-direct]` |
| cache bits | `.BYPASS` + `.LTC256B` | `[VERIFY: sass-derived]` — may not be expressible in the DSL |
| wait encoding | try_wait tight spin | `[VERIFY: sass-derived]` may not be expressible in the DSL [EXPR: A=mbarrier_try_wait; B=-; C=-] |

| Parameter | Value | Source |
|---|---|---|
| bM | 64 | [VERIFY: ncu-direct] grid=8192 |
| prologue warm-up | 2 full tiles | [VERIFY: none] producer leads by 1 tile slot |
"""


def test_parse_gap_queue_reads_the_inline_verify_tag_without_a_verify_column():
    rows = parse_gap_queue(_CANDIDATES_INLINE)
    by = {r.decision: r for r in rows}
    assert set(by) == {"MMA atom", "K-loop schedule", "cache bits", "prologue warm-up"}
    assert by["MMA atom"].reason == "verify-none" and by["MMA atom"].verify == "none"
    # the first tag on the row is the row's channel, a later one is prose
    assert by["K-loop schedule"].reason == "verify-none"
    assert by["cache bits"].reason == "expressibility-guess"
    assert by["prologue warm-up"].reason == "verify-none"
    # a verified row and a guess carrying its [EXPR:] verdict are not gaps
    assert "tile bM x bN x bK" not in by and "wait encoding" not in by and "bM" not in by


def test_parse_gap_queue_bare_column_tolerates_backticks_and_glosses():
    text = ("| Decision | Value | VERIFY | Notes |\n|---|---|---|---|\n"
            "| a | 1 | `none`（only in SASS） | |\n| b | 2 | none (SASS) | |\n"
            "| c | 3 | `[VERIFY: none]` | |\n| d | 4 | `ncu-direct` | |\n")
    assert sorted(r.decision for r in parse_gap_queue(text)) == ["a", "b", "c"]


def test_gap_queue_is_banked_reloaded_and_archived_outside_the_kb(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.recon_path.parent.mkdir()
    ctx.recon_path.write_text("r")
    (tmp_path / "recon" / "candidates.md").write_text(_CANDIDATES, encoding="utf-8")
    ctx.evidence_dir = tmp_path / "evidence"
    # a repo layout: hw_kb/ beside docs/kb-provenance/ -> the human-review archive
    ctx.kb_dir = tmp_path / "repo" / "hw_kb" / "cutedsl"
    ctx.kb_dir.mkdir(parents=True)
    (tmp_path / "repo" / "docs" / "kb-provenance").mkdir(parents=True)
    rows = _persist_gap_queue(ctx)
    assert len(rows) == 3
    banked = (tmp_path / "evidence" / "gap_queue.md").read_text(encoding="utf-8")
    assert "Out-of-run gap queue" in banked and "cache bits" in banked and "DEPBAR.LE 0x1" in banked
    assert "do not enter hw_kb" in banked
    archive = tmp_path / "repo" / "docs" / "kb-provenance" / "gaps" / "demo.md"
    assert archive.read_text(encoding="utf-8") == banked
    assert not list(ctx.kb_dir.iterdir())                   # nothing lands in the KB
    assert [r.decision for r in load_gap_queue(ctx)] == [r.decision for r in rows]
    # a re-affirmed table that closed every row removes both banked copies
    (tmp_path / "recon" / "candidates.md").write_text(
        "| Decision | Value | VERIFY |\n|---|---|---|\n| x | 1 | `ncu-direct` |\n", encoding="utf-8")
    assert _persist_gap_queue(ctx) == []
    assert not (tmp_path / "evidence" / "gap_queue.md").exists() and not archive.exists()
    ctx.recon_path = None
    assert load_gap_queue(ctx) == []


def test_gap_queue_without_a_provenance_dir_banks_only_the_evidence_copy(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.recon_path.parent.mkdir()
    ctx.recon_path.write_text("r")
    (tmp_path / "recon" / "candidates.md").write_text(_CANDIDATES, encoding="utf-8")
    ctx.evidence_dir = tmp_path / "evidence"
    assert len(_persist_gap_queue(ctx)) == 3
    assert (tmp_path / "evidence" / "gap_queue.md").exists()
    assert not (tmp_path.parent / "docs").exists()


def test_gap_queue_alone_never_opens_a_harvest_turn(tmp_path):
    """Unverified recon rows are the reference's structure as narrated — they
    are banked for humans, never handed to the teacher to write into hw_kb."""
    be = FakeBackend(["RECIPE_RECORDED: hw_kb/cutedsl/sm90/never.md"], [])
    ctx = _ctx(tmp_path, be)
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.recon_path.parent.mkdir()
    ctx.recon_path.write_text("r")
    (tmp_path / "recon" / "candidates.md").write_text(_CANDIDATES, encoding="utf-8")
    assert load_gap_queue(ctx)                              # rows exist ...
    assert run_kb_harvest(ctx) == []                        # ... and no turn runs
    assert be.prompts == []
    template = (Path(__file__).parent.parent / "meta" / "kb_harvest_prompt.md").read_text()
    assert "do not enter the KB" in template and "GAP_RECORDED" not in template \
        and "[pending certification]" in template


def test_inferred_rows_join_the_queue_and_reach_the_distill_prompt(tmp_path):
    """A structural reading whose discriminator printed nothing stays
    `[VERIFY: inferred]` (dsa_sparse: the KV-ring stage axis, two
    byte-identical readings, shipped as sass-derived "no ambiguity" and cost a
    13-decision reopen). It is queued, and the distill step is told to carry
    both readings as [ESTIMATE] candidates."""
    from forgekernel.loop.driver import _inferred_rows_block_for_distill
    text = ("| Decision | Value | VERIFY | Notes |\n|---|---|---|---|\n"
            "| KV ring segment axis | 32 key × 512 d or 64 key × 256 d | `inferred` | G1d printed 0 lines |\n"
            "| stage count | 4 | `sass-derived` | immediates 0x0/0x8000/0x10000/0x18000 |\n")
    rows = parse_gap_queue(text)
    assert [(r.decision, r.reason) for r in rows] == [("KV ring segment axis", "inferred")]
    assert "the discriminating observation gave no positive reading" in rows[0].line()
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    recon = tmp_path / "recon"
    recon.mkdir()
    ctx.recon_path = recon / "report.md"
    (recon / "candidates.md").write_text(text, encoding="utf-8")
    block = _inferred_rows_block_for_distill(ctx)
    assert ("Inferred rows" in block and "KV ring segment axis" in block
            and "must not enter the M0 Gate" in block)
    (recon / "candidates.md").write_text(text.replace("`inferred`", "`sass-derived`"),
                                         encoding="utf-8")
    assert _inferred_rows_block_for_distill(ctx) == ""
