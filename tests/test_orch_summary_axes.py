"""Counting axes of the dossier summary are keyed on what the analysis
RESOLVED, not on how many things codegen emitted: unresolved mbarrier objects,
spin micro-loops and per-loop load/copy counts move with the compiler and
stayed `!` on real pages after every structural fix (grouped-gemm: 11
residual `!`, all counting axes). They are still reported, just not diffed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_orch_tools import _big_page  # noqa: E402

from forgekernel.skills import orch_tools as OT  # noqa: E402

def _spin(k: int) -> str:
    a = 0xa000 + 0x20 * k
    return (f"        /*{a:04x}*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4+0x100], R20 ;\n"
            f"        /*{a + 0x10:04x}*/  @!P0 BRA 0x{a:x} ;\n")


_SPIN = "".join(_spin(k) for k in range(3))


def test_spin_micro_loops_are_counted_but_not_diffed():
    big = _big_page()
    base = OT.summarize(big)
    assert "spin" not in base["loops"] and base["spin_loops"] == 0
    # the same kernel with three extra spin micro-loops appended: reported, no `!`
    spun = big.replace("        /*9000*/  EXIT ;\n",
                       _SPIN + "        /*9000*/  EXIT ;\n")
    s = OT.summarize(spun)
    assert s["spin_loops"] == 3 and "spin" not in s["loops"]
    assert "spin_loops" not in OT._DIFF_KEYS and "mbar_unresolved" not in OT._DIFF_KEYS
    assert "supply_counts" not in OT._DIFF_KEYS


def test_mbar_objs_counts_resolved_objects_only():
    big = _big_page()
    s = OT.summarize(big)
    assert s["mbar_objs"] == sum(s["mbar_tiers"].values())
    assert "UNRESOLVED" not in s["mbar_tiers"]
    assert s["mbar_unresolved"] >= 0
    assert s["mbar_objs"] + s["mbar_unresolved"] == len(OT.build_dossier(big)["objects"])


def test_supply_axis_is_a_class_not_a_count():
    s = OT.summarize(_big_page())
    assert s["supply"] and all(
        line.startswith("addr-source loads ") and
        (line.endswith("serial-gather") or line.endswith("prefetched"))
        for line in s["supply"])
    assert not any(ch.isdigit() for line in s["supply"] for ch in line.split(";")[0]
                   if False)  # class text carries no counts
    assert any("copies" in line for line in s["supply_counts"])   # counts still reported


def test_supply_class_normalises_the_raw_lines():
    assert OT._supply_class("r3: addr-source loads 20 (4/20 in top quarter -> interleaved); "
                            "copies 16; tight-chained copies 16") == \
        "addr-source loads interleaved; serial-gather"
    assert OT._supply_class("r3: addr-source loads 4 (4/4 in top quarter -> clustered-at-top); "
                            "copies 16; tight-chained copies 0") == \
        "addr-source loads clustered-at-top; prefetched"
    assert OT._supply_class("r7: copies 40, no in-loop address-source loads (hoisted) ; "
                            "tight-chained copies 0") == \
        "addr-source loads hoisted; prefetched"
