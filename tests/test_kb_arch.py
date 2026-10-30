"""One hw_kb tree per architecture, and the loop enforces which one a run gets.

hw_kb/<language>/ has no shared root: sm90/ and sm100/ are parallel trees
(hw_kb/cutedsl/README.md).
``kb_arch`` (loop.toml [loop].kb_arch / --kb-arch, default sm90) picks the tree,
and ``driver.kb_read_dir`` is the single function the recipe lookup, the three
leak-lint gates, the quarantine and the prompt bindings all go through -- so what
a run may read, what it is linted against and what its prompts name are the
same directory by construction.
"""
import inspect
import re
from pathlib import Path

import pytest

from forgekernel.loop import cli
from forgekernel.loop.cli import _kb_arch_problem, _resolve_kb_arch
from forgekernel.loop.driver import (
    _KB_CERT_MARK,
    _bindings,
    _kb_certify_lint_gate,
    _quarantine_leaky_kb,
    _verify_reopen_price,
    kb_read_dir,
    render_prompt,
)
from forgekernel.loop.kb_lint import lint_kb_dir
from test_loop_driver import REPO, FakeBackend, _ctx

# Every prompt the driver renders with _bindings and that names the KB.
KB_PROMPTS = ("student_prompt.md", "distill_prompt.md", "distill_blackbox_prompt.md",
              "kb_harvest_prompt.md", "kb_recipe_prompt.md", "patch_prompt.md",
              "recon_prompt.md")


def _kb(tmp_path, *arches):
    kb = tmp_path / "hw_kb" / "cutedsl"
    for a in arches:
        (kb / a).mkdir(parents=True)
        (kb / a / "index.md").write_text("# index\n", encoding="utf-8")
    return kb


# --- resolution ----------------------------------------------------------------

def test_kb_arch_defaults_to_sm90_so_older_runs_read_what_they_always_did():
    assert _resolve_kb_arch(None, {}) == "sm90"
    assert _resolve_kb_arch(None, {"loop": {"kb_arch": "sm100"}}) == "sm100"
    assert _resolve_kb_arch("sm100", {"loop": {"kb_arch": "sm90"}}) == "sm100"  # flag wins


def test_kb_arch_must_be_a_plain_directory_name():
    for bad in ("../sm90", "sm90/", "SM100", "hw_kb/cutedsl/sm100", ""):
        if not bad:
            continue
        with pytest.raises(SystemExit):
            _resolve_kb_arch(bad, {})


def test_a_tree_that_does_not_exist_or_has_no_index_refuses_the_launch(tmp_path):
    kb = _kb(tmp_path, "sm90")
    (kb / "sm120").mkdir()  # a directory with no index.md is not a KB tree
    assert _kb_arch_problem(kb, "sm90") is None
    missing = _kb_arch_problem(kb, "sm100")
    assert missing and "no sm100/ tree" in missing and "available: sm90" in missing
    assert "no index.md" in _kb_arch_problem(kb, "sm120")


def test_the_kb_arch_exit_code_means_nothing_else():
    """_main refuses to launch for a dozen reasons, each with its own exit code: a
    watcher reading the code must not have to guess between 'phase products on
    disk' (9) and 'no such KB tree'. The other codes are literals at their return
    sites, so read them from the source rather than from a list that goes stale."""
    src = inspect.getsource(cli)
    taken = {int(m) for m in re.findall(r"^\s+return (\d+)\b", src, re.M)}
    assert {7, 8, 9} <= taken, "the literal return sites moved — update this test"
    assert cli.EXIT_KB_ARCH_MISSING not in taken | {0, cli.EXIT_WINDOW_UNDECLARED}


def test_the_real_trees_are_each_launchable_and_lint_clean():
    """Every hw_kb/cutedsl/<arch>/ with an index.md is a tree a run can be pointed at, and
    each passes the startup gate on its own -- an sm100 run must not be stopped by
    a Hopper finding it can never read, and vice versa."""
    kb = REPO / "hw_kb" / "cutedsl"
    trees = sorted(d.name for d in kb.iterdir() if d.is_dir() and (d / "index.md").is_file())
    assert {"sm90", "sm100"} <= set(trees)
    for arch in trees:
        assert _kb_arch_problem(kb, arch) is None
        assert lint_kb_dir(kb / arch) == [], arch


def test_cli_module_resolves_the_names_its_startup_gate_uses():
    """The startup KB-lint gate runs only in a real launch, so a missing import
    there is invisible to every test that never executes cli._main()."""
    for name in ("kb_read_dir", "lint_kb_dir", "_resolve_kb_arch", "_kb_arch_problem"):
        assert hasattr(cli, name), f"cli.py uses {name} but does not define/import it"


# --- bindings and prompts ----------------------------------------------------

def test_bindings_render_the_runs_tree_from_the_same_function_the_gates_use(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert ctx.kb_arch == "sm90"
    b = _bindings(ctx)
    assert b["{KB_ARCH}"] == "sm90"
    assert b["{KB_READ_DIR}"] == str(kb_read_dir(ctx)) == str(tmp_path / "sm90")
    ctx.kb_arch = "sm100"
    assert _bindings(ctx)["{KB_READ_DIR}"] == str(tmp_path / "sm100")


def test_no_prompt_spells_an_architecture_into_the_kb_path():
    """A Blackwell run used to be told the index was sm90/index.md and to file its
    recipes under sm90/: the prompts had the architecture hard-coded."""
    for name in KB_PROMPTS:
        text = (REPO / "meta" / name).read_text(encoding="utf-8")
        assert "hw_kb/cutedsl/sm90" not in text and "{KB_DIR}/sm90" not in text, name


@pytest.mark.parametrize("name", KB_PROMPTS)
def test_every_kb_prompt_renders_the_runs_tree_and_leaves_no_token_behind(tmp_path, name):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_arch = "sm100"
    out = render_prompt(REPO / "meta" / name, _bindings(ctx))
    assert "{KB_ARCH}" not in out and "{KB_READ_DIR}" not in out, name
    assert "sm100" in out, f"{name} never names the run's tree"
    assert "sm90" not in out, f"{name} still points an sm100 run at Hopper"


def test_write_side_prompts_file_recipes_into_the_runs_tree(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_arch = "sm100"
    for name in ("kb_harvest_prompt.md", "kb_recipe_prompt.md"):
        out = render_prompt(REPO / "meta" / name, _bindings(ctx))
        assert f"{tmp_path}/sm100/index.md" in out, name
        assert f"{tmp_path}/sm100/<interface>.md" in out, name
        assert "hw_kb/cutedsl/sm100/<" in out, name  # the RECIPE_* line the driver parses


# --- lookup, lint, quarantine ------------------------------------------------

def test_a_kb_citation_resolves_only_inside_the_runs_tree(tmp_path):
    kb = _kb(tmp_path, "sm90", "sm100")
    (kb / "sm90" / "ring_depth.md").write_text(
        "# ring_depth\ncertifying probe PROBE PASS max_diff=0.0, job 1\n", encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = kb
    ctx.kb_arch = "sm90"
    assert _verify_reopen_price("reopen_price: +3% via KB:ring_depth", [], [], ctx) is None
    ctx.kb_arch = "sm100"
    why = _verify_reopen_price("reopen_price: +3% via KB:ring_depth", [], [], ctx)
    assert why and "not found under" in why and str(kb / "sm100") in why


def test_a_ten_digit_job_pointer_certifies_an_entry(tmp_path):
    """The Blackwell pool hands out 10-digit job ids where the Hopper pools
    hand out 5-7. An entry certified by a KEEP census cites job pointers and
    nothing else -- no PROBE PASS line -- so a width that stops at 7 turns every
    such sm100 entry into 'carries no GPU certification'."""
    kb = _kb(tmp_path, "sm100")
    (kb / "sm100" / "box_rank.md").write_text(
        "# box_rank\n- certification evidence: census + gate window, job 1789171541 → job 1789171666\n",
        encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = kb
    ctx.kb_arch = "sm100"
    assert _verify_reopen_price("reopen_price: +3% via KB:box_rank", [], [], ctx) is None


def test_every_index_row_marked_verified_is_an_entry_the_lookup_certifies():
    """What an index calls ✅ Verified must be what _verify_reopen_price accepts: a
    teacher citing the index's own word must not be told the entry carries no
    certification. Checked on the status cell, not the whole row -- a transcribed
    row may quote the phrase for one sub-claim."""
    for arch in ("sm90", "sm100"):
        index = (REPO / "hw_kb" / "cutedsl" / arch / "index.md").read_text(encoding="utf-8")
        seen = 0
        for row in index.splitlines():
            m = re.match(r"^\| \[`([a-z0-9_]+)`\]", row)
            status = row.rstrip().rstrip("|").rsplit("|", 1)[-1]
            if not m or "✅ Verified" not in status:
                continue
            seen += 1
            entry = REPO / "hw_kb" / "cutedsl" / arch / f"{m.group(1)}.md"
            assert entry.is_file(), f"{arch}/index.md lists {m.group(1)}, which does not exist"
            assert _KB_CERT_MARK.search(entry.read_text(encoding="utf-8")), (
                f"{arch}/{m.group(1)}.md is ✅ Verified in the index but the lookup would "
                f"reject a KB: citation of it as uncertified")
        assert seen, f"{arch}/index.md has no ✅ Verified row — the table format changed?"


def test_the_certify_lint_gate_sees_only_the_runs_tree(tmp_path):
    kb = _kb(tmp_path, "sm90", "sm100")
    (kb / "sm90" / "leaky.md").write_text("schedule order extracted from the teacher oracle\n", encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = kb
    ctx.kb_arch = "sm100"
    assert _kb_certify_lint_gate(ctx, "d", fix_rounds=0) == []
    ctx.kb_arch = "sm90"
    findings = _kb_certify_lint_gate(ctx, "d", fix_rounds=0)
    assert findings and all(f.startswith("sm90/leaky.md") for f in findings)


def test_quarantine_resolves_a_finding_against_the_kb_root(tmp_path):
    """Findings are relative to the lint root's parent (hw_kb/cutedsl/), so the
    run's tree name is part of the path; the quarantine lands next to hw_kb/,
    outside every tree."""
    kb = _kb(tmp_path, "sm100")
    leaky = kb / "sm100" / "leaky.md"
    leaky.write_text("extracted from the teacher oracle\n", encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = kb
    ctx.kb_arch = "sm100"
    _quarantine_leaky_kb(ctx, ["sm100/leaky.md:1: banned narrative"])
    assert not leaky.exists()
    assert (tmp_path / "kb_quarantine" / "leaky.md").is_file()
    assert (kb / "sm100" / "index.md").is_file()  # the tree itself is untouched
