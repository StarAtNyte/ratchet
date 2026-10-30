"""Tests for NCU profiler parsing + per-iteration metric-evolution trace.

These give the meta agent INDEPENDENT hardware evidence (bank conflicts, register
spill, tensor/warp activity, stalls) per iteration, so it classifies the binding
constraint from measured truth rather than the student's prose self-report.
"""

from __future__ import annotations

from pathlib import Path

from forgekernel.inner.ncu import (
    KEY_METRICS,
    NCU_METRICS_ARG,
    _coerce_dims,
    build_ncu_trace,
    kernel_split,
    key_metrics,
    launch_count,
    parse_ncu_csv,
    profiler_from_ncu,
    render_ncu_reconciliation,
    render_ncu_report,
    render_ncu_trace,
    select_dominant_kernel,
    summarize_metrics,
)
from forgekernel.inner.trajectory import IterationRecord

FIXTURE = Path(__file__).parent / "fixtures" / "ncu_sample.csv"


def _csv() -> str:
    return FIXTURE.read_text()


def test_parse_groups_metrics_by_kernel():
    parsed = parse_ncu_csv(_csv())
    assert len(parsed) == 2  # preprocess + main kernel
    main = next(k for k in parsed if "BwdMainKernel" in k)
    assert parsed[main]["launch__registers_per_thread"] == "168"
    assert parsed[main]["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] == "72746086"


def test_select_dominant_kernel_is_slowest():
    parsed = parse_ncu_csv(_csv())
    dominant = select_dominant_kernel(parsed)
    assert "BwdMainKernel" in dominant  # 3.08ms >> preprocess 117us


def test_key_metrics_extracts_decision_subset_from_dominant():
    km = key_metrics(parse_ncu_csv(_csv()))
    # numeric coercion + short keys
    assert km["regs_per_thread"] == 168
    assert km["bankconf_st"] == 72746086
    assert km["spill_st_sectors"] == 32698880
    assert km["tensor_pct"] == 37.44
    assert km["warp_active_pct"] == 14.87
    assert km["dram_pct"] == 8.99
    # 'n/a' values coerce to None, not a crash
    assert "lts__t_sectors_op_atom_red.sum" not in km
    # non-decision metric is dropped
    assert "sm__inst_executed_pipe_alu.sum" not in km


def test_profiler_from_ncu_is_key_metrics_plus_the_split():
    csv = _csv()
    parsed = parse_ncu_csv(csv)
    assert profiler_from_ncu(csv) == {**key_metrics(parsed), **kernel_split(parsed)}


def test_launch_geometry_comes_from_csv_columns_not_metrics():
    # Block/Grid Size are standard NCU columns, so they are readable from output
    # captured before this mapping existed — no --metrics change, no re-profiling.
    km = key_metrics(parse_ncu_csv(_csv()))
    assert km["block_size"] == 384    # 3 warpgroups x 128
    assert km["grid_size"] == 5120    # dim tuple stored as its product
    assert "Block Size" not in NCU_METRICS_ARG


def test_kernel_split_counts_kernels_and_dominant_share():
    # An M0 that freezes "preprocess / main / postprocess" needs the split itself
    # checked; both numbers are launch facts, not the agent's self-report.
    split = kernel_split(parse_ncu_csv(_csv()))
    assert split["kernel_count"] == 2
    assert split["dominant_time_pct"] == 96.35


def test_kernel_split_on_empty_input_is_empty_not_a_crash():
    assert kernel_split({}) == {}


def test_dim_tuple_coercion():
    assert _coerce_dims("(256, 1, 1)") == 256
    assert _coerce_dims("(32, 16, 10)") == 5120
    assert _coerce_dims("") is None
    assert _coerce_dims("n/a") is None


def test_ncu_metrics_arg_matches_what_the_parser_reads():
    # gpu_entry.sh requests this string; a metric the parser knows but nobody
    # requests is silently absent from every profiler dict.
    assert set(NCU_METRICS_ARG.split(",")) == set(KEY_METRICS)


def test_build_ncu_trace_series_over_iterations():
    recs = [
        IterationRecord(iter=10, decision_id="D5.2", change="cut1", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.700,
                        profiler={"bankconf_st": 80_000_000, "spill_st_sectors": 33_000_000,
                                  "tensor_pct": 36.0, "warp_active_pct": 14.0}),
        IterationRecord(iter=11, decision_id="D5.2", change="cut2", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086, "spill_st_sectors": 32_698_880,
                                  "tensor_pct": 37.44, "warp_active_pct": 14.87}),
    ]
    trace = build_ncu_trace(recs)
    assert [r["iter"] for r in trace] == [10, 11]
    # the evolution is preserved so the meta agent can read the change process
    assert trace[0]["bankconf_st"] == 80_000_000
    assert trace[1]["bankconf_st"] == 72_746_086


def test_build_ncu_trace_skips_iterations_without_profiler():
    recs = [
        IterationRecord(iter=8, decision_id="D3.1", change="x", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.697, profiler={}),
        IterationRecord(iter=11, decision_id="D5.2", change="y", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086}),
    ]
    trace = build_ncu_trace(recs)
    assert [r["iter"] for r in trace] == [11]


def test_render_ncu_trace_is_markdown_table_with_metric_columns():
    recs = [
        IterationRecord(iter=11, decision_id="D5.2", change="y", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086, "tensor_pct": 37.44}),
    ]
    md = render_ncu_trace(recs)
    assert "| iter |" in md
    assert "bankconf_st" in md
    assert "11" in md
    assert "72746086" in md


def test_render_handles_empty_trace():
    md = render_ncu_trace([
        IterationRecord(iter=8, decision_id="D3.1", change="x", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.697, profiler={}),
    ])
    assert "no ncu" in md.lower() or "empty" in md.lower()


# --- NCU summary / report (digest the many metrics into the key parts) -------

def test_summarize_bank_conflict_dominated():
    m = {"bankconf_st": 72_746_086, "spill_st_sectors": 32_698_880,
         "tensor_pct": 37.4, "warp_active_pct": 14.9, "dram_pct": 9.0}
    s = summarize_metrics(m)
    assert "bank" in s["verdict"].lower()
    # bankconf flagged at the top severity
    flagged = {f["key"]: f["severity"] for f in s["flags"]}
    assert flagged.get("bankconf_st") == "severe"


def test_summarize_memory_bound():
    m = {"bankconf_st": 0, "spill_st_sectors": 0, "tensor_pct": 20,
         "warp_active_pct": 60, "dram_pct": 88.0}
    s = summarize_metrics(m)
    assert "memory" in s["verdict"].lower() or "bandwidth" in s["verdict"].lower()


def test_summarize_register_spill_latency():
    m = {"bankconf_st": 0, "spill_st_sectors": 33_000_000, "regs_per_thread": 168,
         "tensor_pct": 30, "warp_active_pct": 15, "dram_pct": 9}
    s = summarize_metrics(m)
    assert "spill" in s["verdict"].lower() or "latency" in s["verdict"].lower()


def test_summarize_clean_no_pathology():
    m = {"bankconf_st": 0, "spill_st_sectors": 0, "tensor_pct": 60,
         "warp_active_pct": 55, "dram_pct": 40}
    s = summarize_metrics(m)
    assert "no dominant" in s["verdict"].lower()


def test_summarize_empty_metrics():
    s = summarize_metrics({})
    assert s["flags"] == [] and "no" in s["verdict"].lower()


# --- wavefront-volume ratio vs the arch ref (the silent scalar-access tax) ----
#
# Numbers from a bwd run: the student's per-element
# copy atoms ran 3.2x the ref's total shared wavefronts while the LOAD side was
# nearly conflict-free — invisible to every absolute threshold above.

_STUDENT_0812 = {"smem_wavefronts_st": 148_837_014, "smem_wavefronts_ld": 82_353_560,
                 "bankconf_st": 0, "tensor_pct": 38.7, "warp_active_pct": 17.7,
                 "dram_pct": 6.0}
_REF_0812 = {"smem_wavefronts_st": 59_232_772, "smem_wavefronts_ld": 13_356_263}


def test_summarize_wavefront_volume_needs_the_baseline():
    # Without the ref signature the ratio flag cannot fire (and must not crash);
    # with conflicts at zero this kernel looks clean — that is the silent tax.
    s = summarize_metrics(_STUDENT_0812)
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}


def test_summarize_wavefront_volume_severe_vs_ref():
    s = summarize_metrics(_STUDENT_0812, baseline=_REF_0812)
    flagged = {f["key"]: f for f in s["flags"]}
    f = flagged["smem_wavefronts_vs_ref"]
    assert f["severity"] == "severe" and f["value"] == 3.18
    assert "atom" in f["note"]
    assert "access-path" in s["verdict"]


def test_summarize_wavefront_outranks_bank_conflicts_in_the_verdict():
    # When both fire, the atom is the cause and the conflicts are its side
    # effect — a bwd run patched the swizzle first and the wall never moved.
    m = dict(_STUDENT_0812, bankconf_st=73_142_086)
    s = summarize_metrics(m, baseline=_REF_0812)
    assert "access-path" in s["verdict"]
    assert {f["key"] for f in s["flags"]} >= {"smem_wavefronts_vs_ref", "bankconf_st"}


def test_summarize_wavefront_ratio_silent_when_baseline_predates_the_metric():
    # Baselines archived before the wavefront columns simply lack the keys.
    s = summarize_metrics(_STUDENT_0812, baseline={"tensor_pct": 55.7})
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}


def test_summarize_wavefront_parity_is_not_flagged():
    m = dict(_REF_0812, tensor_pct=55.7, warp_active_pct=44.0, dram_pct=9.0)
    s = summarize_metrics(m, baseline=_REF_0812)
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}


# --- two-sided reconciliation vs the ref's archived signature -----------------
#
# fp8_gemm-shaped numbers: the student's tensor WORK already matches the ref
# (cycles ratio ~1.000) and the occupancy bubble alone covers ~81% of the time
# residual — the "stop structure knobs" signal the one-sided digest never
# carried.

_RECON_STUDENT = {"tensor_pct": 75.5, "time_ns": 116640,
                  "dram_bytes_read": 96_500_000, "dram_bytes_write": 34_100_000}
_RECON_REF = {"tensor_pct": 81.7, "time_ns": 107744,
              "dram_bytes_read": 95_800_000, "dram_bytes_write": 33_900_000,
              "bankconf_st": 0, "regs_per_thread": 168, "smem_dynamic": 232448}


def test_reconciliation_flags_overlap_bound_when_work_is_isomorphic():
    block = render_ncu_reconciliation(_RECON_STUDENT, _RECON_REF, student_iter=12,
                                      ref_role="architecture", ref_name="deep_gemm")
    assert block is not None
    # the reference identifies itself and the student side names its iter
    assert "[architecture: deep_gemm]" in block
    assert "student(best KEEP, iter 12)" in block
    # the ref side's key values are quoted, not just the ratios
    assert "regs_per_thread=168" in block and "smem_dynamic=232448" in block
    assert "tensor_pct_vs_ref = +6.2 pp" in block
    assert "tensor_cycles_ratio = 1.000" in block
    assert "ncu_time_ratio = 1.083" in block
    assert "OVERLAP-BOUND" in block and "NOT structure" in block


def test_reconciliation_library_ref_suppresses_the_verdict():
    # Same numbers, but the signature is the LIBRARY baseline — a different
    # implementation. The ratios keep their information value; the isomorphism
    # verdict does not, and is replaced by an explicit suppression note.
    block = render_ncu_reconciliation(_RECON_STUDENT, _RECON_REF,
                                      ref_role="library", ref_name="cuBLAS")
    assert block is not None
    assert "[library: cuBLAS]" in block
    assert "tensor_cycles_ratio = 1.000" in block and "ncu_time_ratio = 1.083" in block
    assert "OVERLAP-BOUND" not in block
    assert "isomorphism verdict suppressed" in block


def test_reconciliation_near_parity_gets_no_verdict_line():
    # gemm-shaped: both sides ~93% tensor, times inside the noise band — there
    # is no residual worth attributing, so the block renders WITHOUT a verdict.
    block = render_ncu_reconciliation({"tensor_pct": 93.3, "time_ns": 57050},
                                      {"tensor_pct": 93.5, "time_ns": 57000},
                                      ref_role="architecture")
    assert block is not None
    assert "OVERLAP-BOUND" not in block and "suppressed" not in block


def test_reconciliation_is_omitted_for_non_tensor_refs_and_missing_data():
    # A non-tensor ref (memory-bound op) has nothing to reconcile against.
    assert render_ncu_reconciliation({"tensor_pct": 40.0, "time_ns": 1000},
                                     {"tensor_pct": 0.2, "time_ns": 900}) is None
    # Missing either side degrades to silence, never a crash.
    assert render_ncu_reconciliation(None, _RECON_REF) is None
    assert render_ncu_reconciliation(_RECON_STUDENT, {}) is None
    assert render_ncu_reconciliation({"time_ns": 1000}, _RECON_REF) is None
    assert render_ncu_reconciliation({"tensor_pct": 40.0}, _RECON_REF) is None


def test_render_ncu_report_is_concise_and_flags_key_parts():
    recs = [
        IterationRecord(iter=11, decision_id="D5.2", change="x", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086, "spill_st_sectors": 32_698_880,
                                  "tensor_pct": 37.4, "warp_active_pct": 14.9, "dram_pct": 9.0}),
    ]
    r = render_ncu_report(recs)
    assert "iter 11" in r
    assert "bank" in r.lower()          # the key pathology surfaced
    assert "72746086" in r or "72.7" in r  # the key number shown
    # it is a SUMMARY, not the raw metric dump — verdict line present
    assert "bound" in r.lower() or "pathology" in r.lower()


# --- per-launch aggregation ---------------------------------------------------
#
# NCU emits one row set per LAUNCH. Keeping only the last one (a plain overwrite)
# divides a multi-launch kernel's recorded device time by its launch count, and the
# resulting gap between profiled kernel time and the timed wall clock is
# indistinguishable from a host bottleneck -- so the under-count both hides real
# kernel time AND manufactures false host-bound / mis-aimed-profile verdicts.

_HEADER = ('"ID","Kernel Name","Block Size","Grid Size","Metric Name",'
           '"Metric Unit","Metric Value"')


def _rows(launch_id, kernel, time_ns, regs, conflicts, block="(128, 1, 1)"):
    each = [("gpu__time_active.sum", "ns", time_ns),
            ("launch__registers_per_thread", "register/thread", regs),
            ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "", conflicts)]
    return "\n".join(
        f'"{launch_id}","{kernel}","{block}","(64, 1, 1)","{name}","{unit}","{value}"'
        for name, unit, value in each)


def _multi_launch_csv():
    """One kernel launched 4 times (100 us each) plus a single 150 us setup kernel."""
    lines = [_HEADER]
    for i in range(4):
        lines.append(_rows(i, "chunk_scan_kernel", 100_000, 168, 1_000_000))
    lines.append(_rows(4, "setup_kernel", 150_000, 32, 0))
    return "\n".join(lines) + "\n"


def test_additive_metrics_are_summed_over_launches():
    parsed = parse_ncu_csv(_multi_launch_csv())
    main = parsed["chunk_scan_kernel"]
    assert main["gpu__time_active.sum"] == "400000"          # 4 x 100us, not 100us
    assert main["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] == "4000000"


def test_the_launch_count_travels_with_the_row():
    parsed = parse_ncu_csv(_multi_launch_csv())
    assert launch_count(parsed["chunk_scan_kernel"]) == 4
    assert launch_count(parsed["setup_kernel"]) == 1
    km = key_metrics(parsed)
    assert km["dominant_launches"] == 4
    split = kernel_split(parsed)
    assert split["kernel_count"] == 2 and split["total_launches"] == 5


def test_non_additive_metrics_are_not_summed():
    # Registers per thread is a launch attribute; 4 launches must not read as 672.
    km = key_metrics(parse_ncu_csv(_multi_launch_csv()))
    assert km["regs_per_thread"] == 168


def test_dominance_follows_total_time_not_a_single_launch():
    # 4 x 100us beats one 150us launch. Under last-launch-wins the setup kernel
    # would have been declared dominant and every NCU-anchored gate line would have
    # been compared against it.
    parsed = parse_ncu_csv(_multi_launch_csv())
    assert select_dominant_kernel(parsed) == "chunk_scan_kernel"
    assert kernel_split(parsed)["total_time_ns"] == 550_000


def test_non_additive_metrics_come_from_the_dominant_launch_not_the_last():
    # Launch 0 does the work; launch 1 is a short tail. "Last wins" would record the
    # tail's register footprint as the kernel's.
    csv_text = "\n".join([_HEADER,
                          _rows(0, "k", 900_000, 200, 0),
                          _rows(1, "k", 10_000, 24, 0)]) + "\n"
    parsed = parse_ncu_csv(csv_text)
    km = key_metrics(parsed)
    assert km["regs_per_thread"] == 200
    # The digest is per launch (see the unit section below), so the summed 910_000
    # is reported as its 2-launch mean. The window total stays available where it
    # belongs — the runtime split, which the reconciliation normalises itself.
    assert km["time_ns"] == 455_000
    assert km["dominant_launches"] == 2
    assert kernel_split(parsed)["total_time_ns"] == 910_000


def test_launches_are_separated_without_an_id_column():
    # Some captured CSVs predate / omit the ID column; a repeated metric name then
    # marks the next launch. Falling back to overwrite here would silently restore
    # the very under-count this fixes.
    header = '"Kernel Name","Metric Name","Metric Value"'
    body = '\n'.join(['"k","gpu__time_active.sum","100000"'] * 3)
    parsed = parse_ncu_csv(header + "\n" + body + "\n")
    assert parsed["k"]["gpu__time_active.sum"] == "300000"
    assert launch_count(parsed["k"]) == 3


def test_a_single_launch_kernel_parses_exactly_as_before():
    # The whole existing fixture is single-launch; aggregation must be a no-op on it,
    # or every archived NCU signature becomes incomparable.
    parsed = parse_ncu_csv(_csv())
    main = next(k for k in parsed if "BwdMainKernel" in k)
    assert parsed[main]["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] == "72746086"
    assert launch_count(parsed[main]) == 1


def test_values_with_thousands_separators_are_still_numbers():
    header = '"Kernel Name","Metric Name","Metric Value"'
    body = '"k","dram__bytes_read.sum","1,234,567"'
    parsed = parse_ncu_csv(header + "\n" + body + "\n")
    assert key_metrics(parsed)["dram_bytes_read"] == 1_234_567


def test_a_metric_that_is_na_in_every_launch_stays_absent():
    header = '"ID","Kernel Name","Metric Name","Metric Value"'
    body = "\n".join(['"0","k","gpu__time_active.sum","n/a"',
                      '"1","k","gpu__time_active.sum","n/a"'])
    km = key_metrics(parse_ncu_csv(header + "\n" + body + "\n"))
    assert "time_ns" not in km


# --- unit of the digest: per launch, not window total -------------------------
#
# parse_ncu_csv sums additive counters over launches — that total is what the
# runtime split and the reconciliation need — but every consumer of key_metrics
# reads one launch's worth: _THRESHOLDS bounds are calibrated per launch,
# _wavefront_ratio divides a student digest by an archive profiled through an
# independently configured --launch-count, and milestone gate lines quote archived
# signatures verbatim. Emitting window totals here flips clean kernels to "severe"
# by the launch count alone.

def _n_launch_csv(n, kernel="k", time_ns=100_000, regs=168, conflicts=400_000):
    lines = [_HEADER]
    for i in range(n):
        lines.append(_rows(i, kernel, time_ns, regs, conflicts))
    return "\n".join(lines) + "\n"


def test_key_metrics_reports_additive_counters_per_launch():
    km = key_metrics(parse_ncu_csv(_n_launch_csv(24)))
    assert km["bankconf_st"] == 400_000       # per launch, not 24 x that
    assert km["time_ns"] == 100_000
    assert km["dominant_launches"] == 24
    assert km["regs_per_thread"] == 168       # attributes were never summed


def test_a_clean_multi_launch_kernel_does_not_read_as_a_pathology():
    """400k conflicts per launch is clean (severe line 10M); 24 launches summed is
    9.6M, which would have reported as elevated."""
    s = summarize_metrics(key_metrics(parse_ncu_csv(_n_launch_csv(24))))
    assert "bankconf_st" not in {f["key"] for f in s["flags"]}


def test_wavefront_ratio_is_launch_count_independent():
    """Student and reference are profiled through separate --launch-count settings
    (the operator's own ncu task vs the template's ref_ncu_cmd, 24 by default).

    Before per-launch reporting the ratio scaled by their quotient: the same kernel
    against its own archived signature came out 24x, crossing the 2.0 severe line,
    whose verdict tells the student to rewrite its copy atoms.
    """
    student = key_metrics(parse_ncu_csv(_n_launch_csv(8)))
    ref = key_metrics(parse_ncu_csv(_n_launch_csv(24)))
    s = summarize_metrics(student, baseline=ref)
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}
    assert "copy atom" not in s["verdict"]
