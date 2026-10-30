"""stall_price_run --pm: on-demand PM ride-along, ledger-first fallback."""
import sys
import types

from forgekernel.skills import stall_price_run as spr


def test_build_cmd_adds_pm_sections_only_on_demand():
    plain = spr.build_cmd(pm=False)
    combined = spr.build_cmd(pm=True)
    assert "PmSampling" not in " ".join(plain)
    joined = " ".join(combined)
    assert "--section PmSampling" in joined
    assert "--section PmSampling_WarpStates" in joined
    assert "--pm-sampling-interval 8192" in joined
    # the pricing recipe itself is untouched by the ride-along
    assert plain[0:3] == combined[0:3] == ["ncu", "--set", "full"]
    assert plain[-1] == combined[-1] == "eval/ncu_run.py"
    # the profiler window must open at the kernel, not at process start —
    # without this the launch budget goes to the setup storm and the ledger
    # prices a setup kernel long enough to pass the mis-aim floor (observed in a past job)
    assert "--profile-from-start off" in " ".join(plain)
    assert "--profile-from-start off" in " ".join(combined)


def test_build_cmd_threads_the_pm_interval_knob():
    fast = spr.build_cmd(pm=True, pm_interval=2048)
    assert "--pm-sampling-interval 2048" in " ".join(fast)


def test_collect_gates_on_the_report_not_the_exit_code(monkeypatch, capsys):
    # teardown-crash class: nonzero rc, complete report — must be kept
    monkeypatch.setattr(spr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="LaunchFailed at teardown"))
    monkeypatch.setattr(spr.os.path, "exists", lambda p: True)
    assert spr._collect(pm=False) is True
    assert "teardown-crash class" in capsys.readouterr().out
    # and rc==0 with no report is still a failed collection
    monkeypatch.setattr(spr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(spr.os.path, "exists", lambda p: False)
    assert spr._collect(pm=False) is False


def _fake_tools(monkeypatch, calls):
    st = types.SimpleNamespace(
        price_report=lambda *a, **k: calls.append("price"),
        segment_ledger=lambda *a, **k: calls.append("segment"))
    pt = types.SimpleNamespace(
        timeline_report=lambda *a, **k: calls.append("timeline"))
    monkeypatch.setitem(sys.modules, "stall_tools", st)
    monkeypatch.setitem(sys.modules, "pm_timeline_tools", pt)


def test_combined_failure_falls_back_and_never_costs_the_ledger(
        monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    attempts = []
    monkeypatch.setattr(
        spr, "_collect",
        lambda pm, pm_interval=spr.PM_INTERVAL: (attempts.append(pm), not pm)[1])
    calls = []
    _fake_tools(monkeypatch, calls)
    assert spr.main(["--pm"]) == 0
    out = capsys.readouterr().out
    assert attempts == [True, False]          # combined first, then plain
    assert calls == ["price", "segment"]      # ledger delivered, timeline skipped
    assert "PMTL-FAIL reason=combined-collection-failed" in out
    assert "pm_timeline_run.py" in out        # the standalone retry path is named


def test_combined_success_prices_and_renders_from_one_collection(
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attempts = []
    monkeypatch.setattr(
        spr, "_collect",
        lambda pm, pm_interval=spr.PM_INTERVAL: (attempts.append((pm, pm_interval)),
                                                 True)[1])
    calls = []
    _fake_tools(monkeypatch, calls)
    assert spr.main(["--pm", "--pm-interval", "2048"]) == 0
    assert attempts == [(True, 2048)]         # the knob reaches the collection
    assert calls == ["price", "segment", "timeline"]


def test_without_the_flag_nothing_pm_happens(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    attempts = []
    monkeypatch.setattr(
        spr, "_collect",
        lambda pm, pm_interval=spr.PM_INTERVAL: (attempts.append(pm), True)[1])
    calls = []
    _fake_tools(monkeypatch, calls)
    assert spr.main([]) == 0
    assert attempts == [False]
    assert calls == ["price", "segment"]
    assert "PMTL" not in capsys.readouterr().out


def test_page_rendered_once_and_shared_with_the_dossier(monkeypatch, tmp_path):
    """One `ncu --import` for the page: the segment ledger gets the rendered
    text, and the dossier gets the same text plus the dominant launch index
    (the page carries every captured launch; the dossier must read one)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spr, "_collect",
                        lambda pm, pm_interval=spr.PM_INTERVAL: True)
    page = "0x10  NOP ;\n" * 250
    seen = {}
    st = types.SimpleNamespace(
        price_report=lambda *a, **k: None,
        render_sass_text=lambda p: (page, None),
        dominant_launch=lambda p: (3, "main_kernel"),
        segment_ledger=lambda *a, **k: seen.__setitem__("segment", k))
    ot = types.SimpleNamespace(
        report=lambda text, launch=None: seen.__setitem__("report", (text, launch)))
    monkeypatch.setitem(sys.modules, "stall_tools", st)
    monkeypatch.setitem(sys.modules, "orch_tools", ot)
    assert spr.main([]) == 0
    assert seen["segment"]["sass_text"] is page
    assert seen["report"] == (page, 3)


def test_dossier_degrades_when_the_page_render_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spr, "_collect",
                        lambda pm, pm_interval=spr.PM_INTERVAL: True)
    seen = {}
    st = types.SimpleNamespace(
        price_report=lambda *a, **k: None,
        render_sass_text=lambda p: (None, "`ncu --import` exited 1"),
        dominant_launch=lambda p: (0, "k"),
        segment_ledger=lambda *a, **k: seen.__setitem__("segment", k))
    monkeypatch.setitem(sys.modules, "stall_tools", st)
    monkeypatch.setitem(sys.modules, "orch_tools", types.SimpleNamespace(
        report=lambda *a, **k: seen.__setitem__("report", True)))
    monkeypatch.setattr(spr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="", returncode=1))
    assert spr.main([]) == 0
    assert seen["segment"]["sass_text"] is None      # ledger renders its own
    assert "report" not in seen                      # too small a page: skipped
    assert "SASS page unavailable" in capsys.readouterr().out


def test_role_exposure_rides_the_same_page_and_launch(monkeypatch, tmp_path):
    """The role ledger is the fourth reader of one collection: same rendered
    page and dominant launch as the dossier, after it, never costing the
    ledger when it fails."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spr, "_collect",
                        lambda pm, pm_interval=spr.PM_INTERVAL: True)
    page = "0x10  NOP ;\n" * 250
    seen = {}
    order = []
    st = types.SimpleNamespace(
        price_report=lambda *a, **k: order.append("price"),
        render_sass_text=lambda p: (page, None),
        dominant_launch=lambda p: (3, "main_kernel"),
        segment_ledger=lambda *a, **k: order.append("segment"))
    ot = types.SimpleNamespace(
        report=lambda text, launch=None: order.append("dossier"))
    rt = types.SimpleNamespace(
        report=lambda rep, sass_text=None, launch=None: (
            order.append("roles"), seen.__setitem__("args", (rep, sass_text, launch))))
    monkeypatch.setitem(sys.modules, "stall_tools", st)
    monkeypatch.setitem(sys.modules, "orch_tools", ot)
    monkeypatch.setitem(sys.modules, "role_exposure_tools", rt)
    assert spr.main([]) == 0
    assert order == ["price", "segment", "dossier", "roles"]
    assert seen["args"] == (spr.REPORT + ".ncu-rep", page, 3)


def test_role_exposure_failure_never_costs_the_ledger(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spr, "_collect",
                        lambda pm, pm_interval=spr.PM_INTERVAL: True)
    calls = []
    _fake_tools(monkeypatch, calls)

    def boom(*a, **k):
        raise RuntimeError("no regions")
    monkeypatch.setitem(sys.modules, "role_exposure_tools",
                        types.SimpleNamespace(report=boom))
    assert spr.main([]) == 0
    assert calls == ["price", "segment"]
    assert "role exposure skipped (RuntimeError: no regions)" in capsys.readouterr().out


def test_bundle_restores_a_head_clipped_log():
    from forgekernel.skills import stall_price_run as SP
    head = "=== STALL LEDGER BEGIN ===\nrow 1\n=== STALL LEDGER END ===\n"
    tail = "=== ROLE EXPOSURE BEGIN ===\nROLE label=x\n=== ROLE EXPOSURE END ===\nROLE-OK roles=1\n"
    full = head + tail
    lines = []
    SP.emit_bundle(full, out=lines.append)
    assert lines[0] == SP.BUNDLE_BEGIN and lines[-2] == SP.BUNDLE_END
    assert lines[-1].startswith("BUNDLE-OK bytes=")
    # a capped read kept only the tail: the ledger block is gone from the plain text
    clipped = tail + "\n".join(lines) + "\n"
    assert "STALL LEDGER BEGIN" not in clipped
    expanded = SP.expand_bundle(clipped)
    assert head in expanded and tail in expanded
    # an intact log is returned untouched; garbage is tolerated
    assert SP.expand_bundle(full + "\n".join(lines)) == full + "\n".join(lines)
    assert SP.expand_bundle(SP.BUNDLE_BEGIN + "\n@@@\n" + SP.BUNDLE_END) \
        == SP.BUNDLE_BEGIN + "\n@@@\n" + SP.BUNDLE_END


def test_main_emits_the_bundle_after_the_analysis(monkeypatch, tmp_path, capsys):
    from forgekernel.skills import stall_price_run as SP
    monkeypatch.setattr(SP, "_collect", lambda pm, pm_interval=SP.PM_INTERVAL: True)
    monkeypatch.setattr(SP, "_analyze", lambda st, pm, pmc, *rest: print("=== STALL LEDGER BEGIN ===\nrow\n=== STALL LEDGER END ==="))
    monkeypatch.setattr(SP, "IR_DIR", str(tmp_path / "ir"))
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "stall_tools.py").write_text("")
    monkeypatch.chdir(tmp_path)
    assert SP.main([]) == 0
    out = capsys.readouterr().out
    assert out.index("=== STALL LEDGER END ===") < out.index(SP.BUNDLE_BEGIN)
    clipped = out[out.index(SP.BUNDLE_BEGIN):]          # what a 300-line archive keeps
    assert "STALL LEDGER BEGIN" in SP.expand_bundle(clipped) and "BUNDLE-OK" in out


def test_app_overrides_the_launch_target_only_when_given():
    """`--app` names the profiler-window entry of a tree that is not a student
    workspace (a recon tree: python3 dsa_ref.py ncu); without it the student
    contract eval/ncu_run.py stands."""
    default = spr.build_cmd(pm=True)
    assert default[-2:] == [spr.sys.executable, "eval/ncu_run.py"]
    custom = spr.build_cmd(pm=True, app=["python3", "dsa_ref.py", "ncu"])
    assert custom[-3:] == ["python3", "dsa_ref.py", "ncu"]
    assert custom[:-3] == default[:-2]                  # the recipe is unchanged


def test_main_forwards_app_to_every_collection(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(spr, "_collect",
                        lambda pm, pm_interval=spr.PM_INTERVAL, app=None:
                        (seen.append((pm, app)), not pm)[1])
    _fake_tools(monkeypatch, [])
    assert spr.main(["--pm", "--app", "python3", "dsa_ref.py", "ncu"]) == 0
    assert seen == [(True, ["python3", "dsa_ref.py", "ncu"]),
                    (False, ["python3", "dsa_ref.py", "ncu"])]
    assert spr.main(["--app"]) == 0
    assert "STALL-FAIL reason=bad-args" in capsys.readouterr().out


def test_page_readers_get_the_dominant_kernel_name(monkeypatch, tmp_path):
    """The dossier and the role ledger pick the launch by the dominant action's
    NAME when their copy accepts it (a bystander on the page never stands in);
    an older copy without the argument keeps working on the index."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spr, "_collect", lambda pm, pm_interval=spr.PM_INTERVAL: True)
    page = "0x10  NOP ;\n" * 250
    seen = {}
    st = types.SimpleNamespace(
        price_report=lambda *a, **k: None,
        render_sass_text=lambda p: (page, None),
        dominant_launch=lambda p: (3, "main_kernel"),
        segment_ledger=lambda *a, **k: None)
    ot = types.SimpleNamespace(
        report=lambda text, launch=None, kernel="": seen.__setitem__("dossier", (launch, kernel)))
    rt = types.SimpleNamespace(
        report=lambda rep, sass_text=None, launch=None, kernel_substr="": (
            seen.__setitem__("roles", (launch, kernel_substr))))
    monkeypatch.setitem(sys.modules, "stall_tools", st)
    monkeypatch.setitem(sys.modules, "orch_tools", ot)
    monkeypatch.setitem(sys.modules, "role_exposure_tools", rt)
    assert spr.main([]) == 0
    assert seen == {"dossier": (3, "main_kernel"), "roles": (3, "main_kernel")}


def test_pm_interval_is_read_before_app_only(monkeypatch, tmp_path):
    """Everything after --app belongs to the launch target: a --pm-interval
    there is the app's flag, never this script's (and is not stripped)."""
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(spr, "_collect",
                        lambda pm, pm_interval=spr.PM_INTERVAL, app=None: (
                            calls.append((pm, pm_interval, app)), False)[1])
    assert spr.main(["--pm", "--pm-interval", "2048", "--app", "python3", "ref.py", "ncu"]) == 0
    assert calls[0] == (True, 2048, ["python3", "ref.py", "ncu"])       # first (PM) collection
    calls.clear()
    assert spr.main(["--pm", "--app", "python3", "ref.py", "--pm-interval", "1"]) == 0
    assert calls[0] == (True, spr.PM_INTERVAL, ["python3", "ref.py", "--pm-interval", "1"])


def test_workload_flag_scopes_the_launch_target_of_every_collection(monkeypatch, tmp_path, capsys):
    """A group card's other meta workloads are priced one job each; `--workload`
    (before --app) sets FK_LB_WORKLOAD_SCOPE, which the launch target's ncu_window reads
    and every process ncu starts inherits. A name that is not one token is refused."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(spr.WORKLOAD_SCOPE_ENV, raising=False)
    seen = []
    monkeypatch.setattr(spr, "_collect", lambda pm, pm_interval=spr.PM_INTERVAL, app=None: (
        seen.append((spr.os.environ.get(spr.WORKLOAD_SCOPE_ENV), app)), False)[1])
    assert spr.main(["--pm", "--workload", "wl7", "--app", "python3", "binref.py", "ncu"]) == 0
    assert seen[0] == ("wl7", ["python3", "binref.py", "ncu"])
    assert "workload scope: wl7" in capsys.readouterr().out
    monkeypatch.delenv(spr.WORKLOAD_SCOPE_ENV)
    seen.clear()
    assert spr.main(["--pm", "--app", "python3", "x.py", "--workload", "wl7"]) == 0
    assert seen[0][0] is None                                # after --app: the app's own flag
    assert spr.main(["--pm", "--workload", "wl7;rm"]) == 0
    assert "STALL-FAIL reason=bad-args" in capsys.readouterr().out
