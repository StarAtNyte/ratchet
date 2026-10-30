"""role_exposure_tools: the dossier's regions x the PC-sampled stall ledger x
the tensor plateau, joined per role — on the synthetic two-role kernel
test_orch_tools builds (a copy-producer loop feeding an mma-consumer loop)."""
import re

from forgekernel.skills import orch_tools as OT
from forgekernel.skills import role_exposure_tools as RE
from forgekernel.skills.sass_tools import parse_sass
from test_orch_tools import SASS_SMALL  # tests/ is on sys.path under pytest

LS = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
WAIT = "smsp__pcsamp_warps_issue_stalled_wait"
BAR = "smsp__pcsamp_warps_issue_stalled_barrier"


def _kernel():
    instrs = list(parse_sass(SASS_SMALL).instructions)
    return instrs, OT.find_regions(instrs)


def _per_pc():
    # producer loop [0x100..0x180]: 300 long_scoreboard samples on its LDGSTS
    # consumer loop [0x200..0x340]: 600 wait on the DEPBAR, 100 barrier on BAR.SYNC
    # prologue (straight) [0x10..0x80]: 40 samples; one PC outside the page: 60
    return {0x120: {LS: 200}, 0x130: {LS: 100},
            0x230: {WAIT: 600}, 0x2d0: {BAR: 100},
            0x40: {LS: 40},
            0x9000: {LS: 60}}


def test_role_account_attributes_samples_to_the_dossier_roles():
    instrs, regions = _kernel()
    acc = RE.role_account(instrs, regions, _per_pc())
    assert acc["total"] == 1100 and acc["resolved"] == 1040   # 0x9000 is off-page
    roles = acc["roles"]
    assert roles["copy-producer"]["samples"] == 300
    assert roles["mma-consumer"]["samples"] == 700
    assert roles["straight"]["samples"] == 40
    assert roles["mma-consumer"]["stalls"][WAIT] == 600
    # per-region rows carry the role label so the top-region list reads by role
    labels = {e["label"] for e in acc["regions"].values()}
    assert labels == {"copy-producer", "mma-consumer", "straight"}


def test_exposure_rows_scale_shares_by_the_non_tensor_fraction():
    instrs, regions = _kernel()
    acc = RE.role_account(instrs, regions, _per_pc())
    rows = {r["role"]: r for r in RE.exposure_rows(acc, tensor_active_pct=80.0,
                                                    duration_ns=1_000_000)}
    c = rows["mma-consumer"]
    assert abs(c["share_pct"] - 700 / 1040 * 100) < 1e-6
    # exposed = share x (1 - 0.8): the consumer owns 20% x 67.3% of the kernel
    assert abs(c["exposed_pct"] - 700 / 1040 * 20) < 1e-6
    assert abs(c["exposed_ns"] - 1_000_000 * 700 / 1040 * 0.2) < 1e-3
    # without a tensor reading the shares still print, the estimate does not
    bare = RE.exposure_rows(acc, None, 1_000_000)
    assert all(r["exposed_pct"] is None for r in bare)
    # role order is canonical (producer first), not sample order
    assert [r["role"] for r in bare][:2] == ["copy-producer", "mma-consumer"]


def test_emit_prints_the_block_the_driver_banks_and_a_json_core(capsys):
    instrs, regions = _kernel()
    ok = RE.emit(instrs, regions, _per_pc(), 80.0, "pm-plateau", 1_000_000)
    out = capsys.readouterr().out
    assert ok is True
    assert out.startswith(RE.BEGIN)
    assert re.search(r"^ROLE-OK roles=3 samples=1100 resolved=94\.5 tensor=", out, re.M)
    assert "[tensor] active=80.0% source=pm-plateau duration_ns=1000000" in out
    assert re.search(r"^ROLE label=mma-consumer regions=1 samples=700 share=67\.3 "
                     r"exposed_pct=13\.5 exposed_ns=134615 stalls=wait:86,barrier:14",
                     out, re.M)
    assert re.search(r"^REGION r\d+ copy-producer", out, re.M)
    core = RE.parse_core(out)
    assert core["tensor_active"] == 80.0 and core["tensor_source"] == "pm-plateau"
    assert core["roles"]["copy-producer"]["exposed_ns"] == 57692
    assert core["resolved_pct"] == 94.5


def test_emit_fails_closed_on_an_address_space_mismatch(capsys):
    instrs, regions = _kernel()
    off_page = {0x9000 + 16 * i: {LS: 100} for i in range(10)}
    assert RE.emit(instrs, regions, off_page, 80.0, "pm-plateau", 1) is False
    out = capsys.readouterr().out
    assert "ROLE-FAIL reason=address-space-mismatch" in out
    assert "ROLE-JSON" not in out


def test_align_roles_puts_the_same_role_side_by_side():
    ref = {"tensor_active": 88.0, "tensor_source": "pm-plateau", "duration_ns": 843424,
           "roles": {"copy-producer": {"share_pct": 20.0, "exposed_pct": 2.4,
                                       "exposed_ns": 20242, "stalls": {"long_sb": 90}},
                     "mma-consumer": {"share_pct": 70.0, "exposed_pct": 8.4,
                                      "exposed_ns": 70847, "stalls": {"dispatch": 47}}}}
    stu = {"tensor_active": 83.0, "tensor_source": "pm-plateau", "duration_ns": 874880,
           "roles": {"copy-producer": {"share_pct": 15.0, "exposed_pct": 2.6,
                                       "exposed_ns": 22309, "stalls": {"long_sb": 88}},
                     "mma-consumer": {"share_pct": 80.0, "exposed_pct": 13.6,
                                      "exposed_ns": 118984, "stalls": {"wait": 51}}}}
    table = RE.align_roles(ref, stu)
    head = table.splitlines()[0]
    assert "88.0%" in head and "83.0%" in head
    row = next(l for l in table.splitlines() if l.startswith("mma-consumer"))
    # the reading that decides the locus: consumer exposure +48137 ns, wait vs dispatch
    assert "+48137" in row and "dispatch:47" in row and "wait:51" in row
    prod = next(l for l in table.splitlines() if l.startswith("copy-producer"))
    assert "+2067" in prod
    # a role only one side has prints dashes, not a crash
    stu["roles"]["straight"] = {"share_pct": 5.0, "exposed_pct": 0.9,
                                "exposed_ns": 7000, "stalls": {}}
    assert "straight" in RE.align_roles(ref, stu)


def test_parse_core_tolerates_a_block_without_the_json_line():
    assert RE.parse_core("=== ROLE EXPOSURE BEGIN ===\nROLE-FAIL x\n") is None
    assert RE.parse_core(None) is None


def test_tensor_active_prefers_the_pm_plateau_then_the_aggregate(monkeypatch):
    class _M:
        def __init__(self, v):
            self._v = v

        def value(self):
            return self._v

    class _Action(dict):
        pass

    # no PM series in the report -> the whole-kernel aggregate, labelled as such
    act = _Action()
    act[RE.TENSOR_AGG_METRICS[0]] = _M(55.5)
    monkeypatch.setattr(RE, "_pm_tools", lambda: None)
    assert RE.tensor_active(act) == (55.5, "ncu-aggregate")
    # PM series present -> its plateau wins
    import types
    pmt = types.SimpleNamespace(
        collect_pm_series=lambda a: {"pmsampling:sm__pipe_tensor_cycles_active_realtime"
                                     ".avg.pct_of_peak_sustained_elapsed":
                                     [0, 0, 84, 85, 86, 85, 84, 0]},
        tensor_active_plateau=lambda s: 85.0)
    monkeypatch.setattr(RE, "_pm_tools", lambda: pmt)
    assert RE.tensor_active(act) == (85.0, "pm-plateau")
    # nothing readable -> None, and the source says so
    monkeypatch.setattr(RE, "_pm_tools", lambda: None)
    assert RE.tensor_active(_Action()) == (None, "unavailable")


def test_pm_timeline_tensor_plateau_reads_the_steady_state():
    from forgekernel.skills import pm_timeline_tools as PT
    name = "pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed"
    vals = [0.0] * 5 + [90.0] * 40 + [0.0] * 5
    assert PT.tensor_active_plateau({name: vals}) == 90.0
    assert PT.tensor_active_plateau({"pmsampling:other": [1.0]}) is None


def test_align_roles_prints_each_role_once():
    core = {"tensor_active": 80.0, "tensor_source": "pm-plateau", "duration_ns": 1000,
            "roles": {r: {"share_pct": 1.0, "exposed_pct": 1.0, "exposed_ns": 10,
                          "stalls": {"wait": 100.0}}
                      for r in ("copy-producer", "mma-consumer", "scalar")}}
    table = RE.align_roles(core, core)
    rows = [l for l in table.splitlines() if l.split("|")[0].strip() in
            ("copy-producer", "mma-consumer", "scalar")]
    assert len(rows) == 3          # `a | b - c` once printed every row twice


def test_emit_without_a_tensor_term_is_partial_not_ok(capsys):
    """A bare --set source collection (no PM sections, no tensor aggregate)
    yields the shares but no exposure column; that table must not pass for
    the overlap account — regression: a dsa_sparse ref column of nulls."""
    instrs, regions = _kernel()
    ok = RE.emit(instrs, regions, _per_pc(), None, "unavailable", 1_000_000)
    out = capsys.readouterr().out
    assert ok is True                                   # the shares are banked
    assert re.search(r"^ROLE-PARTIAL roles=3 samples=1100 resolved=94\.5 tensor=none",
                     out, re.M)
    assert not re.search(r"^ROLE-OK", out, re.M)
    assert "stall_price_run.py --pm" in out            # the repair is named
    core = RE.parse_core(out)
    assert core["tensor_active"] is None
    assert all(r["exposed_ns"] is None for r in core["roles"].values())
