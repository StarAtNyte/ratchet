"""Tests for the operator-repo scaffolder (``forge init <op> --kernel-file ...``).

The scaffolder instantiates an isolated operator repo under ``operators/<op>/``
from the tracked main-repo templates: ``src/forgekernel/card_template/`` (workspace docs +
harness skeleton) and ``meta/*_prompt.md`` (the operator-agnostic agent prompts).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from forgekernel.scaffold import new_operator

REPO_ROOT = Path(__file__).resolve().parents[1]

# Runtime placeholders the driver injects per iteration -- the scaffolder MUST
# leave these untouched (only the generate-time {KERNEL_FILE} is substituted).
RUNTIME_PLACEHOLDERS = [
    "{MILESTONE_PATH}",
    "{KB_DIR}",
    "{WORKSPACE}",
    "{STATUS_JSONL}",
    "{TRAJECTORY_PATH}",
    "{META_MILESTONE_PATH}",
    "{DIAGNOSIS_PATH}",
]


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """An isolated main-repo root holding copies of the real templates."""
    shutil.copytree(REPO_ROOT / "meta", tmp_path / "meta")
    shutil.copytree(
        REPO_ROOT / "src" / "forgekernel" / "card_template", tmp_path / "src" / "forgekernel" / "card_template"
    )
    return tmp_path


def test_creates_all_expected_files(base: Path) -> None:
    created = new_operator("demo_op", kernel_file="foo/bar.py", base_dir=base)
    op = base / "operators" / "demo_op"
    expected = [
        op / "README.md",
        op / "window.toml",
        op / "submit.toml",
        op / "submit.local.toml.example",
        op / ".gitignore",
        op / ".gitattributes",
        op / "agent" / "student.md",
        op / "agent" / "diagnose.md",
        op / "agent" / "patch.md",
    ]
    for path in expected:
        assert path.is_file(), f"missing scaffolded file: {path}"
    assert set(created) == set(expected)


def test_op_name_substituted(base: Path) -> None:
    new_operator("demo_op", base_dir=base)
    op = base / "operators" / "demo_op"
    readme = (op / "README.md").read_text(encoding="utf-8")
    assert "demo_op" in readme
    # the template placeholder must be fully consumed
    assert "<op>" not in readme


def test_scaffolded_submit_config_loads_without_credentials(base: Path) -> None:
    """A fresh operator must be able to reach the GPU with no submit.local.toml: the
    scaffolded default is local delivery, and only the branch is the author's to fill."""
    from forgekernel.submit.config import load_config

    new_operator("demo_op", base_dir=base)
    op_dir = base / "operators" / "demo_op"
    text = (op_dir / "submit.toml").read_text(encoding="utf-8")
    assert 'branch = "TODO"' in text
    (op_dir / "submit.toml").write_text(text.replace('branch = "TODO"', 'branch = "main"'),
                                        encoding="utf-8")
    cfg = load_config("demo_op", base_dir=base)
    assert (cfg.delivery, cfg.branch, cfg.exec_gpu, cfg.cluster) == ("local", "main", "0", "local")


def test_no_milestone_is_scaffolded(base: Path) -> None:
    """The milestone is the distill step's output, not a file to be filled in.

    A scaffolded placeholder is a liability: if it ever satisfies the §9 gate,
    _needs_authoring goes False and distillation silently skips, leaving the student
    to optimize against literal placeholders (this has regressed before). Absence cannot drift into looking authored.
    """
    created = new_operator("demo_op", base_dir=base)
    milestone = base / "operators" / "demo_op" / "milestone.md"
    assert not milestone.exists()
    assert milestone not in created


def test_kernel_file_substituted_in_student_prompt(base: Path) -> None:
    new_operator("demo_op", kernel_file="foo/bar.py", base_dir=base)
    student = (base / "operators" / "demo_op" / "agent" / "student.md").read_text(
        encoding="utf-8"
    )
    assert "foo/bar.py" in student
    assert "{KERNEL_FILE}" not in student


def test_runtime_placeholders_preserved(base: Path) -> None:
    new_operator("demo_op", kernel_file="foo/bar.py", base_dir=base)
    agent = base / "operators" / "demo_op" / "agent"
    student = (agent / "student.md").read_text(encoding="utf-8")
    diagnose = (agent / "diagnose.md").read_text(encoding="utf-8")
    # student references the iteration paths; they stay as injection tokens
    for token in ["{MILESTONE_PATH}", "{KB_READ_DIR}", "{KB_ARCH}", "{WORKSPACE}"]:
        assert token in student, f"{token} must survive scaffolding in student.md"
    assert "{TRAJECTORY_PATH}" in diagnose


def test_no_operator_specific_leak_in_meta_prompts() -> None:
    """The committed meta/ prompts must be fully operator-agnostic."""
    student = (REPO_ROOT / "meta" / "student_prompt.md").read_text(encoding="utf-8")
    for leak in ["flash_bwd.py", "flash_attn_dsl", "flash-attn-4", "FA4", "FA3", "cuDNN"]:
        assert leak not in student, f"operator-specific leak in meta prompt: {leak}"


def test_fail_fast_when_operator_exists(base: Path) -> None:
    new_operator("demo_op", base_dir=base)
    with pytest.raises(FileExistsError):
        new_operator("demo_op", base_dir=base)


def test_dispatcher_routes_init_with_options_to_the_scaffold(base: Path) -> None:
    from forgekernel.__main__ import main

    rc = main(["init", "demo_op", "--kernel-file", "foo/bar.py",
               "--base-dir", str(base), "--no-git"])
    assert rc == 0
    assert (base / "operators" / "demo_op" / "agent" / "student.md").is_file()


# --- job payload: the archive must not carry local-only artifacts -------------
#
# Every job extracts `git archive <branch>` fresh. The student and meta agents read the
# workspace directory directly and the driver reads the archives, so none of that needs
# to reach the GPU machine.

def _members(tar: bytes) -> set:
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        return {m.name for m in tf.getmembers() if m.isfile()}


def test_scaffolded_repo_keeps_local_only_artifacts_out_of_the_archive(base: Path) -> None:
    import subprocess

    op = base / "operators" / "demo_op"
    new_operator("demo_op", base_dir=base)
    # the artifacts a live run accumulates, none of which the node opens
    (op / "milestone.md").write_text("x" * 4096, encoding="utf-8")
    (op / "meta_status.jsonl").write_text("{}\n", encoding="utf-8")
    (op / "arch_baseline.json").write_text("{}", encoding="utf-8")
    (op / "student").mkdir(exist_ok=True)
    (op / "student" / "status.jsonl").write_text("{}\n", encoding="utf-8")
    # and the things it does: the kernel, the harness, a probe
    (op / "kernel.py").write_text("k = 1\n", encoding="utf-8")
    (op / "notes").mkdir(exist_ok=True)
    (op / "notes" / "probe_x.py").write_text("p = 1\n", encoding="utf-8")

    run = lambda *a: subprocess.run(["git", "-C", str(op), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    run("add", "-A")
    run("commit", "-q", "-m", "t")
    tar = subprocess.run(["git", "-C", str(op), "archive", "--format=tar", "HEAD"],
                         capture_output=True, check=True).stdout

    shipped = _members(tar)
    for local_only in ("milestone.md", "README.md", "meta_status.jsonl",
                       "arch_baseline.json", "student/status.jsonl",
                       "agent/student.md"):
        assert local_only not in shipped, f"{local_only} must not ride to the node"
    # probes really do run on the node, so notes/ stays
    assert "notes/probe_x.py" in shipped
    assert "kernel.py" in shipped and "submit.toml" in shipped

# ---- git_init sets receive.denyCurrentBranch = updateInstead ----------------


def test_git_init_sets_receive_deny_current_branch(base: Path) -> None:
    """_git_init must configure receive.denyCurrentBranch = updateInstead so that
    the student workspace can push to the card repo's checked-out branch."""
    import subprocess
    from forgekernel.scaffold import _git_init

    new_operator("demo_op", base_dir=base)
    op_dir = base / "operators" / "demo_op"

    # Initialize the repo first so git config has somewhere to write, then set
    # the user identity that the subsequent commit inside _git_init requires.
    subprocess.run(["git", "init", "-q"], cwd=str(op_dir), check=True)
    subprocess.run(["git", "-C", str(op_dir), "config", "user.email", "t@t"],
                   check=True)
    subprocess.run(["git", "-C", str(op_dir), "config", "user.name", "t"],
                   check=True)

    # _git_init re-inits (idempotent), sets the config, then commits.
    _git_init(op_dir, "demo_op")

    result = subprocess.run(
        ["git", "-C", str(op_dir), "config", "receive.denyCurrentBranch"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "updateInstead"


def test_an_uncommitted_gitattributes_does_not_take_effect(base: Path) -> None:
    """Why both over-limit messages say "commit it".

    `git archive <tree-ish>` reads attributes from the ARCHIVED TREE. An operator
    who adds the export-ignore list to the working directory and retries without
    committing gets the identical over-limit failure and no hint as to why — so
    the guidance has to name the commit step, and this pins the behaviour it is
    describing.
    """
    import subprocess

    op = base / "operators" / "demo_op"
    new_operator("demo_op", base_dir=base)
    (op / "milestone.md").write_text("x" * 4096, encoding="utf-8")
    (op / "kernel.py").write_text("k = 1\n", encoding="utf-8")
    attrs = (op / ".gitattributes").read_text(encoding="utf-8")
    (op / ".gitattributes").unlink()

    run = lambda *a: subprocess.run(["git", "-C", str(op), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    run("add", "-A")
    run("commit", "-q", "-m", "no attributes yet")

    def shipped():
        tar = subprocess.run(["git", "-C", str(op), "archive", "--format=tar", "HEAD"],
                             capture_output=True, check=True).stdout
        return _members(tar)

    assert "milestone.md" in shipped()          # nothing excluded yet
    (op / ".gitattributes").write_text(attrs, encoding="utf-8")
    assert "milestone.md" in shipped()          # written but NOT committed: inert
    run("add", "-A")
    run("commit", "-q", "-m", "commit the attributes")
    assert "milestone.md" not in shipped()      # committed: now it bites
