"""The CUDA mode (loop language cuda-sol, a kernel.cu student), kept apart from the CuTe
DSL mode: which prompts / KB / provenance each mode reads (loop/lang_mode.py), the
language-vs-kernel-file guard, and the tools that read or rebuild the kernel — the
timeline probe's C++ markers, the lever-pricing prebuild and round budget, the census
over the card's own build, the job-env toolchain check — plus the DSL mode staying
exactly as it was."""

from __future__ import annotations

import os
import re
import sys
import time
import types
from pathlib import Path

import pytest

from forgekernel.loop import agent_policy as ap
from forgekernel.loop import cli as loop_cli
from forgekernel.loop import driver
from forgekernel.loop import lang_mode
from forgekernel.skills import job_env_check, lever_pricing_run, orch_census_run
from forgekernel.skills import timeline_probe as tp
from test_loop_driver import FakeBackend, _ctx
from test_leaderboard_cuda_student import STUDENT, card_lb  # noqa: F401  (fixture)
from test_binref import base, sol_root, _no_network  # noqa: F401  (fixtures card_lb needs)

REPO = Path(__file__).resolve().parents[1]

MARKED = STUDENT.replace(
    "#include <torch/extension.h>\n",
    "#include <torch/extension.h>\n#define FK_SEG(...) ((void)0)\n").replace(
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n",
    '    FK_SEG("aux.entry");\n'
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    for (int ci = 0; ci < 4; ++ci) {\n"
    '        FK_SEG("state.C.wait_kk.begin", ci);\n'
    '        FK_SEG("state.C.wait_kk.end", ci);\n'
    "        FK_SEG(NAMES[ci], ci);\n"
    '        FK_SEG("state.E.end", ci + 1);\n'
    "    }\n")


# ---------------------------------------------------------------- modes
def test_a_language_belongs_to_one_mode_and_each_mode_reads_its_own_kb_and_provenance(tmp_path):
    assert lang_mode.mode_of("cuda-sol") == lang_mode.MODE_CUDA
    for lang in ("cutedsl", "triton", "inline-ptx", None):
        assert lang_mode.mode_of(lang) == lang_mode.MODE_CUTEDSL
    assert lang_mode.kb_root_name("cutedsl") == "hw_kb/cutedsl"
    assert lang_mode.kb_root_name("cuda-sol") == "hw_kb/cuda"
    assert lang_mode.kb_repo_root(tmp_path / "hw_kb" / "cuda") == tmp_path
    assert lang_mode.provenance_dir(tmp_path, "cutedsl") == tmp_path / "docs" / "kb-provenance"
    assert lang_mode.provenance_dir(tmp_path, "cuda-sol") == tmp_path / "docs" / "kb-provenance" / "cuda"
    cu, dsl, gaps = ("docs/kb-provenance/cuda/x.md", "docs/kb-provenance/x.md",
                     "docs/kb-provenance/gaps/op.md")
    assert lang_mode.owns_provenance(cu, "cuda-sol") and not lang_mode.owns_provenance(dsl, "cuda-sol")
    assert lang_mode.owns_provenance(dsl, "cutedsl") and lang_mode.owns_provenance(gaps, "cutedsl")
    assert not lang_mode.owns_provenance(cu, "cutedsl")


def test_the_cuda_mode_reads_its_prompt_overlay_and_falls_back_to_the_shared_prompt(tmp_path):
    (tmp_path / "cuda").mkdir()
    for n in ("student_prompt.md", "diagnose_prompt.md"):
        (tmp_path / n).write_text("shared")
    (tmp_path / "cuda" / "student_prompt.md").write_text("cuda")
    assert lang_mode.prompt_path(tmp_path, "student_prompt.md", "cuda-sol") == tmp_path / "cuda" / "student_prompt.md"
    assert lang_mode.prompt_path(tmp_path, "student_prompt.md", "cutedsl") == tmp_path / "student_prompt.md"
    assert lang_mode.prompt_path(tmp_path, "diagnose_prompt.md", "cuda-sol") == tmp_path / "diagnose_prompt.md"


def test_the_language_follows_the_kernel_file_and_a_mismatch_is_named():
    assert loop_cli._resolve_language(None, {}, "kernel.cu") == "cuda-sol"
    assert loop_cli._resolve_language(None, {}, "kernel.py") == "cutedsl"
    assert loop_cli._resolve_language(None, {}) == "cutedsl"
    assert loop_cli._resolve_language(None, {"loop": {"language": "cutedsl"}}, "kernel.cu") == "cutedsl"
    assert lang_mode.language_problem("cuda-sol", "kernel.cu") is None
    assert lang_mode.language_problem("cutedsl", "kernel.py") is None
    assert "CUDA mode" in lang_mode.language_problem("cuda-sol", "kernel.py")
    assert "cuda-sol" in lang_mode.language_problem("cutedsl", "kernel.cu")


def test_the_language_mismatch_exit_code_means_nothing_else():
    import inspect
    src = inspect.getsource(loop_cli)
    taken = {int(m) for m in re.findall(r"^\s+return (\d+)\b", src, re.M)}
    codes = {v for k, v in vars(loop_cli).items() if k.startswith("EXIT_") and isinstance(v, int)}
    assert loop_cli.EXIT_LANGUAGE_MISMATCH == 18
    assert loop_cli.EXIT_LANGUAGE_MISMATCH not in taken
    assert sum(1 for v in codes if v == loop_cli.EXIT_LANGUAGE_MISMATCH) == 1


def _tokens(text: str) -> set:
    return set(re.findall(r"\{[A-Z][A-Z_]*\}", text))


def _sections(text: str) -> list:
    return [ln for ln in text.splitlines() if ln.startswith("## ")]


def test_every_cuda_prompt_overlay_keeps_the_shared_prompts_bindings_and_sections():
    """An overlay is its own prompt (the CUDA mode's), but the driver renders it with
    the shared prompt's bindings and parses the same output: a binding the overlay
    dropped, or one the shared prompt gained since, is a silent difference. The
    shared prompt is the reference here; when it changes, the overlay follows."""
    overlays = sorted((REPO / "meta" / "cuda").glob("*.md"))
    assert {p.name for p in overlays} >= {
        "student_prompt.md", "recon_prompt.md", "distill_prompt.md", "distill_blackbox_prompt.md",
        "patch_prompt.md", "kb_recipe_prompt.md", "kb_harvest_prompt.md", "verdict_probe_brief.md"}
    # bindings an overlay drops on purpose: the only {KB_ARCH} of these two pointed at
    # the DSL's ptx_escape entry, which a CUDA student does not need
    waived = {"patch_prompt.md": {"{KB_ARCH}"}, "recon_prompt.md": {"{KB_ARCH}"}}
    for ov in overlays:
        shared = REPO / "meta" / ov.name
        assert shared.is_file(), f"meta/cuda/{ov.name} overlays no shared prompt"
        a, b = shared.read_text(encoding="utf-8"), ov.read_text(encoding="utf-8")
        assert _tokens(b) == _tokens(a) - waived.get(ov.name, set()), (ov.name, _tokens(a) ^ _tokens(b))
        assert _sections(b) == _sections(a), ov.name


def test_the_cuda_prompts_teach_cuda_not_the_dsl():
    dsl = re.compile(r"cutlass\.cute|cute\.compile|CUTE_DSL|inspect\.signature|ptx_escape|"
                     r"make_tiled_copy|const_expr|DSL spelling|DSL package source")
    for ov in sorted((REPO / "meta" / "cuda").glob("*.md")):
        hits = [ln for ln in ov.read_text(encoding="utf-8").splitlines() if dsl.search(ln)]
        assert not hits, (ov.name, hits[:3])
    assert "hw_kb/cuda/{KB_ARCH}/" in (REPO / "meta" / "cuda" / "kb_recipe_prompt.md").read_text()
    assert "docs/kb-provenance/cuda/" in (REPO / "meta" / "cuda" / "kb_recipe_prompt.md").read_text()
    assert "load_cuda_entry" in (REPO / "meta" / "cuda" / "student_prompt.md").read_text()


def test_the_cuda_kb_is_its_own_launchable_lint_clean_tree():
    from forgekernel.loop.kb_lint import lint_kb_dir
    kb = REPO / "hw_kb/cuda"
    assert (kb / "TEMPLATE.md").is_file() and (kb / "README.md").is_file()
    assert loop_cli._kb_arch_problem(kb, "sm100") is None
    assert lint_kb_dir(kb / "sm100") == []
    assert "hw_kb/cuda has no sm90/ tree" in loop_cli._kb_arch_problem(kb, "sm90")
    for entry in sorted((kb / "sm100").glob("*.md")):
        text = entry.read_text(encoding="utf-8")
        assert not re.search(r"\bcute\.[A-Za-z_]|cutlass\.cute|CUTE_DSL|@cute", text), entry.name
        if entry.name != "index.md":
            # nothing is carried over uncertified: every entry says where its readings came from
            assert "Certification source" in text, entry.name


# ---------------------------------------------------------------- driver wiring
def _cuda_ctx(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.language, ctx.kernel_file = "cuda-sol", "kernel.cu"
    return ctx


def test_the_driver_routes_a_cuda_task_to_its_prompts_and_its_kernel_file(tmp_path):
    dsl = _ctx(tmp_path, FakeBackend([], []))
    assert driver._meta_prompt(dsl, "student_prompt.md") == REPO / "meta" / "student_prompt.md"
    assert driver._timeline_cmd(dsl) == driver._TIMELINE_CMD
    cu = _cuda_ctx(tmp_path)
    assert driver._meta_prompt(cu, "student_prompt.md") == REPO / "meta" / "cuda" / "student_prompt.md"
    assert driver._timeline_cmd(cu) == f"{driver._TIMELINE_CMD} --kernel kernel.cu"
    cu.orch_census = True
    dsl.orch_census = True
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / driver._ORCH_CENSUS_RUNNER).write_text("")
    assert driver._ncu_mode(dsl) == driver._ORCH_CENSUS_CMD
    assert driver._ncu_mode(cu) == f"{driver._ORCH_CENSUS_CMD} --kernel kernel.cu"
    assert "/cuda" in driver._bindings(cu)["{PROVENANCE_DIR}"]
    assert not driver._bindings(dsl)["{PROVENANCE_DIR}"].endswith("cuda")


def test_the_cuda_student_timeline_note_teaches_the_cpp_marker(tmp_path):
    ctx = _cuda_ctx(tmp_path)
    ctx.timeline_probe = "on"
    ctx.evidence_dir = tmp_path / "evidence"
    (tmp_path / "kernel.cu").write_text(MARKED)
    note = driver._timeline_student_note(ctx)
    assert "#define FK_SEG(...) ((void)0)" in note and "def _seg" not in note
    assert "cutlass.const_expr" not in note
    assert "`aux.entry`" in note and "`state.E.end`" in note
    assert "FK_SEG(NAMES[ci], ci)" in note       # the computed name is named
    (tmp_path / "kernel.cu").write_text(MARKED.replace("#define FK_SEG(...) ((void)0)\n", ""))
    assert "lacks the `#define FK_SEG` line" in driver._timeline_student_note(ctx)


# ---------------------------------------------------------------- timeline probe (C++)
def test_cpp_markers_are_read_rewritten_and_the_readout_is_bound_into_the_module():
    assert tp.markers(MARKED, cuda=True) == ["aux.entry", "state.C.wait_kk.begin",
                                              "state.C.wait_kk.end", "state.E.end"]
    assert tp.has_marker_def(MARKED, cuda=True) and not tp.has_marker_def(MARKED)
    assert tp.unparsed_markers(MARKED, cuda=True) == ["FK_SEG(NAMES[ci], ci)"]
    stamped, sites = tp.stamp_cuda_source(MARKED, "all", stride=2)
    assert sites == tp.markers(MARKED, cuda=True)
    assert "#define FK_SEG" not in stamped and "FK_SEG(" not in stamped
    assert "_fk_tl_seg(0, (long long)(0))" in stamped and "_fk_tl_seg(3, (long long)(ci + 1))" in stamped
    assert stamped.count("((void)0);") == 1           # the computed name is a no-op
    assert "#define FK_TL_STRIDE 2" in stamped and "%%clock64" in stamped and "%%gridid" in stamped
    assert f"#define FK_TL_NSITE {tp.NSITE}" in stamped and "@NSLOT@" not in stamped
    # the readout sits before the module and is bound inside it, after the student's own defs
    assert stamped.index("static void _fk_tl_read") < stamped.index("PYBIND11_MODULE")
    assert re.search(r'PYBIND11_MODULE\(TORCH_EXTENSION_NAME, m\) \{\n    m\.def\("_fk_tl_read", '
                     r'&_fk_tl_read\); m\.def\("run", &run\); \}', stamped)
    # same stamping twice: no nonce (the slab is the module's own symbol)
    assert tp.stamp_cuda_source(MARKED, "all", stride=2)[0] == stamped
    lean, kept = tp.stamp_cuda_source(MARKED, "waits")
    assert kept == ["state.C.wait_kk.begin", "state.C.wait_kk.end"] and lean.count("((void)0);") == 3
    with pytest.raises(ValueError, match="#define FK_SEG"):
        tp.stamp_cuda_source(MARKED.replace("#define FK_SEG(...) ((void)0)\n", ""))
    with pytest.raises(ValueError, match="PYBIND11_MODULE"):
        tp.stamp_cuda_source(MARKED.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)", "void x()"))


def test_the_stamped_cuda_kernel_passes_the_cards_cuda_fence(card_lb):
    lbm, problem, card, _ = card_lb
    (card / "kernel.cu").write_text(MARKED)
    assert lbm.kernel_fence(problem) is None
    (card / "kernel.cu").write_text(tp.stamp_cuda_source(MARKED, "all", stride=3)[0])
    assert lbm.kernel_fence(problem) is None
    problem = dict(problem, reference_impl=dict(problem["reference_impl"], kind="sealed"))
    assert lbm.kernel_fence(problem) is None


def test_a_cuda_timeline_runs_the_stamped_bench_and_launcher_under_cuda_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "kernel.cu").write_text(MARKED)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ncu_run.py").write_text("")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    monkeypatch.setattr(tp, "run_bench", lambda label, env=None: {
        "correctness": {"pass": True}, "performance": {"device_median_ms": 0.5}})
    seen = {}

    def stamped_bench(label, env=None, timeout=2400, cmd=None):
        seen["bench"] = cmd
        seen["kernel"] = (tmp_path / "kernel.cu").read_text()
        return {"correctness": {"pass": False, "error": "x"}, "performance": None}, [], 0
    monkeypatch.setattr(tp, "run_bench_sampling_clock", stamped_bench)
    assert tp.main(["--mode", "waits", "--kernel", "kernel.cu"]) == 0
    assert seen["bench"].endswith("timeline_probe.py --cuda-run eval/bench.py --mode all")
    assert "_fk_tl_seg(0, (long long)(ci))" in seen["kernel"]
    assert (tmp_path / "kernel.cu").read_text() == MARKED       # restored
    assert "reason=stamped-bench-failed" in capsys.readouterr().out
    # --dry stamps without a compiler
    assert tp.main(["--dry", "--kernel", "kernel.cu"]) == 0
    assert "[dry] stamped CUDA source built" in capsys.readouterr().out


def test_cuda_run_captures_the_cards_build_and_runs_the_script_as_main(tmp_path, monkeypatch):
    built = types.SimpleNamespace(_fk_tl_read=lambda dst: None)
    ext = types.ModuleType("torch.utils.cpp_extension")
    ext.load = lambda *a, **k: built
    utils = types.ModuleType("torch.utils")
    utils.cpp_extension = ext
    torch = types.ModuleType("torch")
    torch.utils = utils
    for name, mod in (("torch", torch), ("torch.utils", utils), ("torch.utils.cpp_extension", ext)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv("TL_SLAB_OUT", raising=False)
    (tmp_path / "eval").mkdir()
    out = tmp_path / "ran.txt"
    (tmp_path / "eval" / "bench.py").write_text(
        "import sys, torch.utils.cpp_extension as e\n"
        f"open({str(out)!r}, 'w').write(' '.join(sys.argv[1:]) + '|' + str(e.load() is not None) "
        "+ '|' + sys.path[0])\n")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert tp.main([tp.CUDA_RUN_FLAG, "eval/bench.py", "--mode", "all"]) == 0
    args, loaded, path0 = out.read_text().split("|")
    assert args == "--mode all" and loaded == "True" and path0 == str(tmp_path / "eval")


def test_the_dsl_timeline_path_is_untouched():
    """The C++ branch is an addition: a kernel.py task stamps and runs as before."""
    src = ('def _seg(site, row=0):\n    return None\n\n\ndef k():\n'
           '    _seg("a.b.begin", 1)\n    _seg("a.b.end", 1)\n')
    assert tp.markers(src) == ["a.b.begin", "a.b.end"] and tp.has_marker_def(src)
    stamped, _ = tp.stamp_source(src)
    assert "import cutlass as _tl_cutlass" in stamped and "_fk_tl_seg" not in stamped
    assert not tp.is_cuda_kernel("kernel.py") and tp.is_cuda_kernel("kernel" + ".cu")


# ---------------------------------------------------------------- lever pricing (C++)
def test_the_prebuild_builds_every_variant_under_the_kernels_own_name(tmp_path, monkeypatch):
    (tmp_path / "eval").mkdir()
    log = tmp_path / "built.txt"
    (tmp_path / "eval" / "leaderboard.py").write_text(
        "import os\n"
        "def load_cuda_entry(path, entry):\n"
        "    src = open(path).read()\n"
        "    if 'BAD' in src:\n        raise RuntimeError('nvcc failed')\n"
        f"    open({str(log)!r}, 'a').write(os.path.basename(path) + ':' + src + '\\n')\n"
        "    raise AttributeError(entry)\n")
    monkeypatch.setattr(lever_pricing_run, "ROOT", str(tmp_path))
    ok = lever_pricing_run.prebuild_cuda({"base": "A", "+x": "B", "+bad": "BAD"}, "kernel.cu", workers=2)
    assert ok == {"base": True, "+x": True, "+bad": False}
    assert sorted(log.read_text().split()) == ["kernel.cu:A", "kernel.cu:B"]


def test_the_timed_rounds_stop_when_the_job_clock_cannot_cover_another(monkeypatch):
    env = {"TASK_TIMEOUT": "1800"}
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    t[0] = 1000.0
    assert lever_pricing_run.round_fits(0.0, 500.0, clock=clock, env=env)       # 800 - 180 >= 500
    assert not lever_pricing_run.round_fits(0.0, 700.0, clock=clock, env=env)   # 800 - 180 < 700
    assert lever_pricing_run.round_fits(0.0, 1e9, clock=clock, env={})          # no clock: no cut


def test_a_comment_only_variant_reads_the_same_sass():
    """nvcc hashes the source into the anonymous namespace's name, so two builds
    whose code is identical still differ in every mangled symbol."""
    def dump(ns, path, op="IADD3"):
        return ("\nFatbin elf code:\n================\narch = sm_100a\nidentifier = " + path + "\n\n"
                "\tcode for sm_100a\n"
                f"\t\tFunction : _ZN41_GLOBAL__N__{ns}_9_kernel_cu_d698c71f5k_bigE14CUtensorMap_st\n"
                f"        /*0000*/                   LDC R1, c[0x0][0x37c] ;  /* 0x0000df00ff017b82 */\n"
                f"        /*0010*/                   {op} R2, R2, 0x1, RZ ;  /* 0x0000000102027810 */\n")
    from forgekernel.skills.lever_pricing_run import sass_code_text
    a = sass_code_text(dump("c5a99a67", "/tmp/fk_kernel_build/fk_kernel_aaa/kernel.cu"))
    b = sass_code_text(dump("0f3e9b21", "/tmp/fk_kernel_build/fk_kernel_bbb/kernel.cu"))
    assert a == b and "identifier" not in a and "LDC R1" in a
    assert sass_code_text(dump("0f3e9b21", "x", op="IMAD")) != a


def test_the_sass_sha_of_a_cuda_variant_is_none_without_its_build(tmp_path, monkeypatch):
    monkeypatch.setenv("FK_KERNEL_BUILD_DIR", str(tmp_path))
    assert lever_pricing_run.cuda_sass_sha("A", "kernel.cu") is None


# ---------------------------------------------------------------- census (C++)
def test_the_census_finds_the_build_of_exactly_this_source(tmp_path, monkeypatch):
    monkeypatch.setenv("FK_KERNEL_BUILD_DIR", str(tmp_path))
    for key, src in (("aaa", "A"), ("bbb", "B")):
        d = tmp_path / f"fk_kernel_{key}"
        d.mkdir()
        (d / "kernel.cu").write_text(src)
        (d / f"fk_kernel_{key}.so").write_bytes(b"\x7fELF")
    assert orch_census_run.find_cuda_build("B", "kernel.cu") == str(tmp_path / "fk_kernel_bbb" / "fk_kernel_bbb.so")
    assert orch_census_run.find_cuda_build("C", "kernel.cu") is None


def test_a_cuda_census_without_a_build_skips_on_one_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FK_KERNEL_BUILD_DIR", str(tmp_path / "none"))
    (tmp_path / "kernel.cu").write_text("x")
    monkeypatch.setattr(orch_census_run, "_run_ncu", lambda env: (0, "csv\n"))
    assert orch_census_run.main(["--kernel", "kernel.cu"]) == 0
    assert "ORCH-CENSUS-SKIP reason=no-cuda-build kernel=kernel.cu" in capsys.readouterr().out


def test_the_seeded_tools_never_name_a_cpp_source_the_recon_guard_refuses():
    from forgekernel import binref
    rx = re.compile("|".join(binref.REF_TREE_FORBID))
    from forgekernel.skills import stall_price_run
    for mod in (tp, lever_pricing_run, orch_census_run, job_env_check, stall_price_run):
        hits = [ln for ln in Path(mod.__file__).read_text().splitlines() if rx.search(ln)]
        assert not hits, (mod.__name__, hits[:3])


# ---------------------------------------------------------------- job env, policy, card
def test_the_cuda_toolchain_check_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(job_env_check.shutil, "which", lambda n: None)
    assert "nvcc" in job_env_check.cuda_toolchain_problem({})
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvcc").write_text("")
    monkeypatch.setitem(sys.modules, "ninja", types.ModuleType("ninja"))
    env = {"CUDA_HOME": str(tmp_path), "CUTLASS_DIR": str(tmp_path / "cutlass")}
    assert "CUTLASS" in job_env_check.cuda_toolchain_problem(env)
    hdr = tmp_path / "cutlass" / "include" / "cutlass"
    hdr.mkdir(parents=True)
    (hdr / ("cutlass" + ".h")).write_text("")
    assert job_env_check.cuda_toolchain_problem(env) is None


def test_the_cuda_students_notice_names_headers_not_the_dsl(tmp_path):
    cfg = {"loop": {"cuda_source_dirs": [str(tmp_path)], "dsl_source_dirs": ["/elsewhere"]}}
    pol = ap.from_config(cfg, oracle_mode="blackbox", repo_root=tmp_path, teacher_trees=[],
                         language="cuda-sol")
    note = pol.notice(ap.ROLE_STUDENT)
    assert "CUDA / CUTLASS headers" in note and "DSL package source" not in note and str(tmp_path) in note
    dsl = ap.from_config(cfg, oracle_mode="blackbox", repo_root=tmp_path, teacher_trees=[])
    assert "DSL package source" in dsl.notice(ap.ROLE_STUDENT) and "/elsewhere" in dsl.notice(ap.ROLE_STUDENT)
    warns = ap.declaration_warnings({"loop": {}}, None, language="cuda-sol")
    assert any("cuda_source_dirs is empty" in w for w in warns)


def test_a_cuda_card_suggests_the_cuda_mode():
    from forgekernel import leaderboard as lb
    prob = types.SimpleNamespace(reference_impl={"kind": "baseline_solution", "name": "r"},
                                 workload={"uuid": "u", "axes": {}}, name="n", group=None,
                                 suite="other", board=None)
    cu = lb.render_loop_toml_example(prob, "op", "kernel.cu")
    py = lb.render_loop_toml_example(prob, "op")
    assert 'language    = "cuda-sol"         # what' in cu
    assert 'language    = "cutedsl"          # what' in py


def test_an_unchanged_cuda_source_is_not_restaged(card_lb, monkeypatch):
    """A fresh mtime on the build's source copy makes ninja recompile an unchanged kernel."""
    lbm, problem, card, _ = card_lb
    ext = types.ModuleType("torch.utils.cpp_extension")
    seen = []
    ext.load = lambda **kw: (seen.append(kw["sources"][0]), types.SimpleNamespace(run=1))[1]
    utils = types.ModuleType("torch.utils")
    utils.cpp_extension = ext
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", ext)
    monkeypatch.setattr(sys.modules["torch"], "utils", utils, raising=False)
    monkeypatch.setenv("FK_KERNEL_BUILD_DIR", str(card.parent / "build"))
    lbm.load_cuda_entry(card / "kernel.cu", "run")
    staged = Path(seen[0])
    old = time.time() - 3600
    os.utime(staged, (old, old))
    lbm.load_cuda_entry(card / "kernel.cu", "run")
    assert staged.stat().st_mtime == pytest.approx(old)


def test_a_lineinfo_build_is_its_own_and_the_timed_build_never_carries_it(card_lb, monkeypatch):
    """The stall ledger's run (FK_CUDA_LINEINFO=1) builds with -lineinfo under fk_lineinfo_*;
    the timed build keeps the board's flags under fk_kernel_* (the census reads only those)."""
    lbm, problem, card, _ = card_lb
    ext = types.ModuleType("torch.utils.cpp_extension")
    seen = []
    ext.load = lambda **kw: (seen.append(kw), types.SimpleNamespace(run=1))[1]
    utils = types.ModuleType("torch.utils")
    utils.cpp_extension = ext
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", ext)
    monkeypatch.setattr(sys.modules["torch"], "utils", utils, raising=False)
    monkeypatch.setenv("FK_KERNEL_BUILD_DIR", str(card.parent / "build"))
    monkeypatch.delenv("FK_CUDA_LINEINFO", raising=False)
    lbm.load_cuda_entry(card / "kernel.cu", "run")
    monkeypatch.setenv("FK_CUDA_LINEINFO", "1")
    lbm.load_cuda_entry(card / "kernel.cu", "run")
    monkeypatch.setenv("FK_CUDA_LINEINFO", "0")
    lbm.load_cuda_entry(card / "kernel.cu", "run")
    timed, li, again = seen
    assert "-lineinfo" not in timed["extra_cuda_cflags"] and timed["name"].startswith("fk_kernel_")
    assert "-lineinfo" in li["extra_cuda_cflags"] and li["name"].startswith("fk_lineinfo_")
    assert again["name"] == timed["name"]


def test_the_stall_run_asks_for_lineinfo_and_the_role_pass_does_not(monkeypatch):
    from forgekernel.skills import stall_price_run
    src = Path(stall_price_run.__file__).read_text()
    assert 'os.environ.setdefault("FK_CUDA_LINEINFO", "1")' in src
    calls = []
    monkeypatch.setattr(lever_pricing_run, "sh",
                        lambda cmd, timeout=0, env=None: (calls.append(env), (0, "", ""))[1])
    lever_pricing_run.run_price("base", timeout=5)
    assert calls[0]["FK_CUDA_LINEINFO"] == "0"


def test_a_build_error_larger_than_a_pipe_does_not_stall_the_prebuild(tmp_path, monkeypatch, capsys):
    """stderr used to be a pipe nobody read: a CUTLASS-sized error blocked the builder until
    the prebuild's timeout."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "leaderboard.py").write_text(
        "import sys\n"
        "def load_cuda_entry(path, entry):\n"
        "    sys.stderr.write('error: template instantiation\\n' * 12000)\n"
        "    sys.stderr.write('error: THE LAST LINE\\n')\n"
        "    raise RuntimeError('nvcc failed')\n")
    monkeypatch.setattr(lever_pricing_run, "ROOT", str(tmp_path))
    t0 = time.monotonic()
    ok = lever_pricing_run.prebuild_cuda({"base": "A"}, "kernel.cu", timeout=30, workers=1)
    assert ok == {"base": False} and time.monotonic() - t0 < 15
    out = capsys.readouterr().out
    assert "PRICE PREBUILD base failed" in out and "  | error: THE LAST LINE" in out
    assert not any(ln.startswith("error:") for ln in out.splitlines())


def test_the_prebuild_never_takes_more_than_its_share_of_the_job_clock():
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    assert lever_pricing_run.prebuild_budget(0.0, clock=clock, env={}) == lever_pricing_run.CUDA_PREBUILD_TIMEOUT
    assert lever_pricing_run.prebuild_budget(0.0, clock=clock, env={"TASK_TIMEOUT": "1800"}) == 720.0
    t[0] = 1700.0
    assert lever_pricing_run.prebuild_budget(0.0, clock=clock, env={"TASK_TIMEOUT": "1800"}) == \
        lever_pricing_run.PREBUILD_MIN_S


def test_a_cuda_role_pass_leaves_the_ablations_their_time():
    env = {"TASK_TIMEOUT": "1800"}
    t = [1300.0]
    clock = lambda: t[0]  # noqa: E731
    reserve = lever_pricing_run.ablation_reserve_s(4, 2, [10.0, 12.0, 11.0])
    assert reserve == pytest.approx(11.0 * 5 * 3)
    # 500 s left: one more 200 s --pm fits under the margin alone (320 s), not with the reserve (155 s)
    assert lever_pricing_run.role_pass_plan(0.0, [200.0], clock=clock, env=env)[0]
    assert not lever_pricing_run.role_pass_plan(0.0, [200.0], clock=clock, env=env, reserve_s=reserve)[0]
    assert lever_pricing_run.ablation_reserve_s(0, 2, [10.0]) == 0.0
    assert lever_pricing_run.ablation_reserve_s(4, 2, []) == 0.0
    assert lever_pricing_run.ablation_reserve_s(4, 2, [999.0]) == lever_pricing_run.ABLATION_RESERVE_CAP_S
