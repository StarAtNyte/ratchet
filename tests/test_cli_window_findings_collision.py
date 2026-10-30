"""The loop CLI imports two different ``window_findings``: the 200k context-window
gate (``loop.agentctx``, takes ``AgentCmdFacts``) and the timed-window contract
check (``window_contract``, takes ``WindowContract``). A merge once let
the second import shadow the first, so the 200k gate called the contract checker
on ``AgentCmdFacts`` and every ``loop start`` died with
``AttributeError: 'AgentCmdFacts' object has no attribute 'reference_inputs'``
(seen at a gdn_prefill_bwd run's preflight). The CLI must bind each under its own name."""

from forgekernel.loop import agentctx, cli
from forgekernel import window_contract


def test_cli_binds_both_window_findings_under_distinct_names():
    assert cli.context_window_findings is agentctx.window_findings
    assert cli.window_findings is window_contract.window_findings
    assert cli.context_window_findings is not cli.window_findings


def test_context_gate_accepts_agent_cmd_facts(tmp_path):
    env = {"PATH": "/usr/bin"}
    facts = agentctx.inspect_agent_cmd("claude -p --model claude-opus-5[1m]", env, tmp_path)
    # Must not raise: the gate reads AgentCmdFacts, not a WindowContract.
    assert isinstance(cli.context_window_findings(facts), list)


def test_dry_run_block_does_not_rebind_the_contract_state():
    """The dry-run plan prints two window lines: the 200k context gate and the
    timed-window contract. The merged CLI once stored the 200k text in
    ``window_state`` — the variable that already held the contract verdict — so
    the contract line rendered 'UNDECLARED — 0 finding(s)' on a declared card.
    Exactly one binding of ``window_state`` in ``_main``: the contract's."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cli._main))
    bindings = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "window_state"
        and isinstance(node.ctx, ast.Store)
    ]
    assert len(bindings) == 1, [b.lineno for b in bindings]
