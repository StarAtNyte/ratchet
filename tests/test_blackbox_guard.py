"""The closed-source run's in-job tripwire (scripts/blackbox-guard.sh).

Two properties, both learned the hard way (a broken guard can burn a run's whole
reconnaissance budget):

* a refusal must be LOUD — the predecessor died inside `set -euo pipefail` before
  printing anything, so every trip looked like the probe itself crashing;
* stdlib a probe legitimately needs (tempfile for an .ncu-rep) must not trip it.
"""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blackbox-guard.sh"


def run_guard(tmp_path: Path, cmd: str, extra: str = "", notes: dict | None = None):
    """Source the guard the way a run's gpu_entry.sh does, under the same shell
    options, and call it on `cmd`."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(exist_ok=True)
    for name, text in (notes or {}).items():
        (notes_dir / name).write_text(text)
    script = (
        "set -euo pipefail\n"
        f'W={tmp_path}\n'
        + (f'BLACKBOX_EXTRA={extra!r}\n' if extra else "")
        + f'source {SCRIPT}\n'
        f'blackbox_guard "$1"\n'
        'echo GUARD-PASSED\n'
    )
    return subprocess.run(["bash", "-c", script, "bash", cmd],
                          capture_output=True, text=True)


def test_clean_probe_passes(tmp_path):
    res = run_guard(tmp_path, "python3 notes/sched_probe.py",
                    notes={"sched_probe.py": "import sass_tools\n"})
    assert res.returncode == 0, res.stderr
    assert "GUARD-PASSED" in res.stdout


def test_no_notes_dir_at_all_is_not_an_error(tmp_path):
    """An empty tree must not look like a violation (nor kill the job)."""
    res = run_guard(tmp_path, "nvidia-smi")
    assert res.returncode == 0, res.stderr
    assert "GUARD-PASSED" in res.stdout


def test_notes_with_no_shell_scripts_does_not_kill_the_job(tmp_path):
    """The regression: an unmatched notes/*.sh glob used to abort the whole job
    inside pipefail, before the refusal was ever printed."""
    res = run_guard(tmp_path, "cat closed/ref.enc",
                    notes={"probe.py": "print(1)\n"})
    assert res.returncode == 3, (res.returncode, res.stdout, res.stderr)
    assert "BLACKBOX-GUARD REFUSED" in res.stdout
    assert "GUARD-PASSED" not in res.stdout


def test_refusal_names_the_matched_tokens(tmp_path):
    res = run_guard(tmp_path, "cat closed/ref.enc")
    assert res.returncode == 3
    assert "closed/" in res.stdout and ".enc" in res.stdout


def test_a_probe_file_trips_it_too(tmp_path):
    res = run_guard(tmp_path, "python3 notes/sneaky.py",
                    notes={"sneaky.py": "import inspect; inspect.getsource(ref)\n"})
    assert res.returncode == 3
    assert "notes:" in res.stdout and "sneaky.py" in res.stdout


def test_the_template_closed_reference_is_on_the_base_list(tmp_path):
    """The shared closed_ref.py loader and its payload are named by every
    self-evolution run's bench; a probe that names them is reaching for the
    reference's source, not its behaviour."""
    for cmd in ("python3 -c 'import closed_ref; print(closed_ref.unseal)'",
                "cat eval/ref_bin/ref_payload.bin",
                "python3 -c 'import os; print(os.environ[\"FK_REF_KEY\"])'"):
        res = run_guard(tmp_path, cmd)
        assert res.returncode == 3, cmd
        assert "BLACKBOX-GUARD REFUSED" in res.stdout


def test_stdlib_tempfile_does_not_trip_it(tmp_path):
    """An NCU probe needs a temp file for its .ncu-rep. Banning tempfile refused
    four of a run's own probes and protected nothing."""
    probe = ("import tempfile\n"
             "with tempfile.NamedTemporaryFile(suffix='.ncu-rep') as t:\n"
             "    pass\n")
    res = run_guard(tmp_path, "python3 notes/ncu_probe.py",
                    notes={"ncu_probe.py": probe})
    assert res.returncode == 0, res.stdout
    assert "GUARD-PASSED" in res.stdout


def test_sass_stays_open(tmp_path):
    """SASS is what black-box reconnaissance IS; only line-resolved source is out."""
    res = run_guard(tmp_path, "ncu --print-source sass --set full python3 ref.py")
    assert res.returncode == 0, res.stdout


def test_line_resolved_source_is_refused(tmp_path):
    res = run_guard(tmp_path, "ncu --print-source cuda python3 ref.py")
    assert res.returncode == 3
    assert "BLACKBOX-GUARD REFUSED" in res.stdout


def test_run_specific_names_via_extra(tmp_path):
    res = run_guard(tmp_path, "python3 -c 'import kda_gdn_ref'", extra="kda_gdn|gdn_core")
    assert res.returncode == 3
    assert "kda_gdn" in res.stdout


def test_extra_is_optional(tmp_path):
    res = run_guard(tmp_path, "python3 -c 'import kda_gdn_ref'")
    assert res.returncode == 0, res.stdout


def test_an_extra_hit_says_so_and_a_base_hit_does_not(tmp_path):
    """BLACKBOX_EXTRA must not contain a substring of the reference kernel's symbol
    name: a gdn run listed `gdn_core` while its main kernel was
    `kernel_cutlass_kernel_gdn_core...`, so a probe filtering on the kernel — and
    every later round, via the resident earlier probe in notes/ — would have been
    refused. The refusal names the run-specific origin so the LIST gets fixed."""
    res = run_guard(tmp_path, "ncu -k regex:gdn_core python3 ref.py", extra="kda_gdn|gdn_core")
    assert res.returncode == 3
    assert "BLACKBOX_EXTRA" in res.stdout and "kernel's symbol name" in res.stdout
    res = run_guard(tmp_path, "cat closed/ref.enc", extra="kda_gdn|gdn_core")
    assert res.returncode == 3
    assert "BLACKBOX_EXTRA" not in res.stdout


def test_a_resident_earlier_probe_is_named_in_the_refusal(tmp_path):
    """notes/ is scanned cumulatively: the file that trips the guard may be one
    the teacher wrote rounds ago. The refusal must point at it."""
    res = run_guard(tmp_path, "python3 notes/round7.py", extra="gdn_core",
                    notes={"recon_01_sass_full.py": "KERNEL = 'kernel_cutlass_kernel_gdn_core'\n",
                           "round7.py": "print('clean')\n"})
    assert res.returncode == 3
    assert "recon_01_sass_full.py" in res.stdout and "round7.py" not in res.stdout.split("matched:")[0]


# --- the production shape: seeded into notes/ and sourced FROM there -------------------
#
# Every test above sources the guard from scripts/. In a run it is not: the driver
# seeds it into the reference tree's notes/ and gpu_entry.sh sources that copy — the same
# notes/ the guard scans. Its own comments and forbid list carry every forbidden token,
# so the seeded copy refused EVERY `cmd` in two Hopper runs before any probe ran,
# and none of the tests above could see it.

def run_seeded(tmp_path: Path, cmd: str, extra: str = "", notes: dict | None = None,
               tree: dict | None = None):
    """Seed the guard into <tree>/notes/ and source that copy, as gpu_entry.sh does."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(exist_ok=True)
    (notes_dir / "blackbox-guard.sh").write_bytes(SCRIPT.read_bytes())
    for name, text in (notes or {}).items():
        (notes_dir / name).write_text(text)
    for name, text in (tree or {}).items():
        (tmp_path / name).write_text(text)
    script = (
        "set -euo pipefail\n"
        f'W={tmp_path}\n'
        + (f'BLACKBOX_EXTRA={extra!r}\n' if extra else "")
        + 'source "$W/notes/blackbox-guard.sh"\n'
        'blackbox_guard "$1"\n'
        'echo GUARD-PASSED\n'
    )
    return subprocess.run(["bash", "-c", script, "bash", cmd],
                          capture_output=True, text=True)


def test_the_seeded_guard_does_not_refuse_itself(tmp_path):
    """The regression: a clean probe, guard sourced from notes/, must pass."""
    res = run_seeded(tmp_path, "python3 notes/r01_census.py",
                     notes={"r01_census.py": "import sass_tools\nprint('census')\n"})
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "GUARD-PASSED" in res.stdout


def test_the_seeded_guard_still_refuses_a_real_probe(tmp_path):
    """Skipping itself must not blind it: a probe with a forbidden token is refused,
    and the refusal names that probe — not the guard."""
    res = run_seeded(tmp_path, "python3 notes/peek.py",
                     notes={"peek.py": "open('closed/ref.enc','rb').read()\n"})
    assert res.returncode == 3, (res.returncode, res.stdout)
    assert "BLACKBOX-GUARD REFUSED" in res.stdout
    named = res.stdout.split("matched:")[0]
    assert "notes/peek.py" in named and "blackbox-guard.sh" not in named


def test_a_probe_cannot_hide_under_the_guards_file_name(tmp_path):
    """Only a byte-identical copy of the RUNNING guard is skipped. A different file named
    blackbox-guard.sh in notes/ is scanned like any probe (here the running guard is the
    one in scripts/, so the notes/ file is just a file with that name)."""
    res = run_guard(tmp_path, "true",
                    notes={"blackbox-guard.sh": "cat closed/ref.enc\n"})
    assert res.returncode == 3, (res.returncode, res.stdout)
    assert "notes/blackbox-guard.sh" in res.stdout


def test_extra_naming_the_trees_own_wrapper_warns_at_source_time(tmp_path):
    """Regression: EXTRA listed `gdn_ls_ref`, the tree's own wrapper that the README
    told probes to import. Say so the moment the guard is sourced — before any probe."""
    res = run_seeded(tmp_path, "nvidia-smi", extra="gdn_ls_ref|fkref_gdn",
                     tree={"gdn_ls_ref.py": "def make_inputs(): ...\n"})
    assert "BLACKBOX-GUARD WARNING" in res.stdout
    assert "gdn_ls_ref" in res.stdout.split("WARNING")[1]
    assert res.returncode == 0          # a warning, not a refusal: nvidia-smi is clean


def test_that_collision_turns_a_refusal_into_a_configuration_error(tmp_path):
    res = run_seeded(tmp_path, "python3 notes/r02.py", extra="gdn_ls_ref|fkref_gdn",
                     notes={"r02.py": "import gdn_ls_ref\nrun = gdn_ls_ref.ref_runner()\n"},
                     tree={"gdn_ls_ref.py": "def ref_runner(): ...\n"})
    assert res.returncode == 3
    assert "run configuration error" in res.stdout
    assert "do not work around it" in res.stdout


def test_extra_that_names_only_sealed_things_does_not_warn(tmp_path):
    res = run_seeded(tmp_path, "nvidia-smi", extra="gdn_ls_ref_blob|fkref_gdn",
                     tree={"gdn_ls_ref.py": "def make_inputs(): ...\n"})
    assert "BLACKBOX-GUARD WARNING" not in res.stdout
    assert res.returncode == 0


def test_every_driver_seeded_tool_passes_the_guard(tmp_path):
    """The driver seeds its toolkits into notes/ — the directory the guard scans before
    every job of a closed-source run — so one forbidden token in a seeded file refuses
    every job of that tree, not one probe. timeline_probe.py imported the stdlib
    byte-compile module, a listed token: Blackwell reference trees refused every
    job from the moment it was seeded, ending reconnaissance with its structure
    reads waived. Seed all of them, as the driver does, and
    run a sanctioned probe."""
    import importlib

    from forgekernel.loop.driver import PRICING_SEEDED_MODULES, RECON_SEEDED_MODULES
    notes = {}
    for name in dict.fromkeys(RECON_SEEDED_MODULES + PRICING_SEEDED_MODULES):
        path = Path(importlib.import_module(name).__file__)
        notes[path.name] = path.read_text()
    assert "timeline_probe.py" in notes and "sass_tools.py" in notes
    res = run_seeded(tmp_path, "python3 notes/r01_census.py",
                     notes={**notes, "r01_census.py": "import sass_tools\n"})
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "GUARD-PASSED" in res.stdout
