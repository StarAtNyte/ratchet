"""The precision gate, the loop's side: the card's contract (loop/precision_gate.py), the
driver applying it to the eval JSON's ``precision`` block, and the launch preflight. The
measure itself is tests/test_precision_measure.py (needs torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import NCU, REC, FakeBackend, _ctx  # noqa: E402

from forgekernel.inner.trajectory import load_records  # noqa: E402
from forgekernel.loop import cli, precision_gate as pg  # noqa: E402
from forgekernel.loop.driver import run_inner_iteration  # noqa: E402
from forgekernel.loop.parse import parse_eval  # noqa: E402



# --- the loop's contract (no torch) ----------------------------------------------------

def _card(tmp_path, *, leaderboard=False, helper=True, bench=None, card_contract=None):
    card = tmp_path / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    if leaderboard:
        (card / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(
            {} if card_contract is None else {"precision_contract": card_contract}))
    if helper:
        (card / "eval" / "precision.py").write_text("# helper\n")
    if bench is not None:
        (card / "eval" / "bench.py").write_text(bench)
    return card


def test_contract_defaults_by_card_kind(tmp_path):
    lbcard = _card(tmp_path, leaderboard=True)
    assert pg.resolve_contract({}, lbcard) == ("off", True)          # the board's rules only
    assert pg.resolve_contract({}, tmp_path / "plain") == ("strict", True)
    assert pg.resolve_contract({"loop": {"precision_contract": "Strict"}}, lbcard) == ("strict", False)
    on = _card(tmp_path / "on", leaderboard=True, card_contract="official")
    assert pg.resolve_contract({}, on) == ("official", True)          # the card turned it on
    bad = _card(tmp_path / "bad", leaderboard=True, card_contract="lenient")
    with pytest.raises(ValueError, match="precision_contract"):
        pg.resolve_contract({}, bad)


def test_a_misspelt_contract_is_an_error():
    with pytest.raises(ValueError, match="precision_contract"):
        pg.resolve_contract({"loop": {"precision_contract": "lenient"}}, Path("."))


def test_bench_findings_want_a_recorder_call_not_a_mention(tmp_path):
    called = "from precision import PrecisionRecorder\nprec = PrecisionRecorder()\n"
    mentioned = "# PrecisionRecorder()\nX = 'PrecisionRecorder()'\n"
    assert pg.bench_findings(_card(tmp_path / "a", bench=called)) == []
    assert any("never calls" in f for f in pg.bench_findings(_card(tmp_path / "b", bench=mentioned)))
    assert any("precision.py is missing" in f
               for f in pg.bench_findings(_card(tmp_path / "c", helper=False, bench=called)))
    # a leaderboard card's bench is generated: the helper has to be there and its
    # problem.json has to turn the measure on
    assert pg.bench_findings(_card(tmp_path / "d", leaderboard=True, card_contract="strict")) == []
    assert pg.bench_findings(_card(tmp_path / "e", leaderboard=True, helper=False,
                                   card_contract="strict"))
    assert any("measures nothing" in f
               for f in pg.bench_findings(_card(tmp_path / "f", leaderboard=True)))


def _block(status="pass", ratio=1.1, stress=None):
    row = {"case": "round0", "output": "C", "dtype": "float16", "e_ref": 2e-4,
           "e_cand": 2e-4 * ratio, "limit": 1e-3, "ratio": ratio, "status": status}
    rows = [row] + ([{**row, "case": "stress", "stress": s, "status": st}
                     for s, st in (stress or {}).items()])
    return {"schema": 1, "status": status, "truth": "float64", "k": 4.0, "floor_ulps": 2.0,
            "n_checks": len(rows), "n_failed": int(status == "fail"), "worst": row,
            "rows": rows, **({"why": "round0 C: e_cand ..."} if status == "fail" else {})}


def test_summarize_and_gate_failure():
    s = pg.summarize("strict", _block("fail", 9.0, {"offset": "pass", "amp": "unavailable"}))
    assert s["status"] == "fail" and s["worst"]["ratio"] == 9.0
    assert s["stress"] == {"offset": "pass", "amp": "unavailable"}
    why = pg.gate_failure("strict", s)
    assert why.startswith("precision gate (strict)") and "x9" in why
    assert pg.gate_failure("official", pg.summarize("official", _block("fail", 9.0))) is None
    assert pg.summarize("off", _block()) == {}
    assert pg.summarize("strict", None) == {"contract": "strict", "status": "unreported"}
    assert "precision=pass" in pg.describe(pg.summarize("official", _block()))


def _eval(block, ok=True):
    obj = {"correctness": {"pass": ok}, "performance": {"bwd_median_ms": 2.4,
                                                        "device_median_ms": 2.4}}
    if block is not None:
        obj["precision"] = block
    return "=== JSON ===\n" + json.dumps(obj)


def test_parse_eval_carries_the_block():
    assert parse_eval(_eval(_block())).precision["status"] == "pass"
    assert parse_eval(_eval(None)).precision is None


def test_strict_contract_reverts_a_failed_precision_check(tmp_path):
    be = FakeBackend([REC("D1.1", 0)], [_eval(_block("fail", 9.0)), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.precision_contract = "strict"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "REVERT" and rec.correctness is False
    assert "precision gate (strict)" in rec.error
    assert rec.precision["status"] == "fail" and rec.precision["contract"] == "strict"
    assert load_records(ctx.status_path)[-1].precision["worst"]["ratio"] == 9.0


def test_official_contract_records_and_keeps(tmp_path, capsys):
    be = FakeBackend([REC("D1.1", 0)], [_eval(_block("fail", 9.0)), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.precision_contract = "official"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP" and rec.precision["status"] == "fail"
    out = capsys.readouterr().out
    assert "precision (official, recorded only)" in out and "precision=fail" in out


def test_a_bench_without_the_block_is_recorded_unreported_and_warned_once(tmp_path, capsys):
    be = FakeBackend([REC("D1.1", 0), REC("D1.2", 0)], [_eval(None), NCU, _eval(None), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.precision_contract = "strict"
    r1 = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    r2 = run_inner_iteration(ctx, 2, metric_before=2.4, last_good_ref=r1.sha, green=True)
    assert r1.verdict == r2.verdict == "KEEP"
    assert r1.precision == {"contract": "strict", "status": "unreported"}
    assert capsys.readouterr().out.count("carries no `precision` block") == 1


def test_strict_with_a_reporting_bench_fails_an_eval_without_the_block(tmp_path, capsys):
    """The bench is wired to report it (launch preflight found nothing missing): a missing
    block is something the kernel could have caused, not an old bench -- a correctness
    failure under strict, recorded only under official."""
    be = FakeBackend([REC("D1.1", 0)], [_eval(None), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.precision_contract = "strict"
    ctx.precision_bench_reports = True
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "REVERT" and rec.correctness is False
    assert pg.MISSING_BLOCK in rec.error
    assert rec.precision == {"contract": "strict", "status": "unreported"}
    be = FakeBackend([REC("D1.1", 0)], [_eval(None), NCU])
    (tmp_path / "o").mkdir()
    ctx = _ctx(tmp_path / "o", be)
    ctx.precision_contract = "official"
    ctx.precision_bench_reports = True
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP"


def test_off_contract_does_not_read_it(tmp_path):
    be = FakeBackend([REC("D1.1", 0)], [_eval(_block("fail", 9.0)), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP" and rec.precision == {}


def test_preflight_refuses_a_fresh_strict_start_without_the_block(tmp_path, capsys):
    card = _card(tmp_path, bench="print('judge')\n")
    rc = cli._precision_preflight(card, "strict", True, resuming=False, dry_run=False)
    assert rc == cli.EXIT_PRECISION_UNREPORTED == 20
    assert "never calls" in capsys.readouterr().err
    assert cli._precision_preflight(card, "strict", True, resuming=True, dry_run=False) == 0
    assert cli._precision_preflight(card, "official", False, resuming=False, dry_run=False) == 0
    assert cli._precision_preflight(card, "off", False, resuming=False, dry_run=False) == 0
    ok = _card(tmp_path / "ok", bench="from precision import PrecisionRecorder\nPrecisionRecorder()\n")
    assert cli._precision_preflight(ok, "strict", True, resuming=False, dry_run=False) == 0
    assert "precision_contract = strict (default for a own bench)" in capsys.readouterr().out


def test_preflight_leaderboard_default_off_is_silent(tmp_path, capsys):
    card = _card(tmp_path, leaderboard=True)
    contract, defaulted = pg.resolve_contract({}, card)
    assert cli._precision_preflight(card, contract, defaulted, resuming=False, dry_run=False) == 0
    out = capsys.readouterr()
    assert "precision_contract = off (default for a leaderboard card)" in out.out
    assert "neither measured nor read" in out.out and out.err == ""
    # a loop that asks for strict over a card whose bench measures nothing is refused
    assert cli._precision_preflight(card, "strict", False, resuming=False,
                                    dry_run=False) == cli.EXIT_PRECISION_UNREPORTED


def test_exit_20_names_nothing_else():
    taken = {v for k, v in vars(cli).items() if k.startswith("EXIT_")
             and k != "EXIT_PRECISION_UNREPORTED"}
    assert cli.EXIT_PRECISION_UNREPORTED not in taken
