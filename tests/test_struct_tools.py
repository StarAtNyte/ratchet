"""struct_tools: the per-segment construction table read off SASS + per-PC
execution counts — segmentation by recurring named barrier, who-does-what by
exec ratio, tile granule by LDSM/STSM immediates, precision by MMA suffix,
hand-off by store/load offset overlap; fails closed on what it cannot read."""
import csv
import io

from forgekernel.loop.driver import _STRUCT_OK
from forgekernel.skills import struct_tools as S
from forgekernel.skills.sass_tools import parse_sass

BAR = 100          # the region barrier: 25 trips x 4 cooperating warps (one CTA)
FILL = 520         # prologue length: past sass_tools' instruction floor


def _kernel(*, seg2_exec=BAR // 2, loop=False, copies=1, barriers=True, loop_trips=40,
            arv=False):
    """(sass text, {addr: exec}) — a 4-warp, 3-segment construction: seg 1 stores
    8x8 tiles (STSM, 0x800 apart) in fp16; seg 2 is a 2-warp shuffle reduction;
    seg 3 loads two tiles 0x80 apart, two HMMA.1688, stores the product.
    ``arv`` makes the region barrier an arrive/sync split on 8 warps (id 0xd,
    0x100 threads) where only the 4 sync-side warps run the segments."""
    lines, counts = [], {}
    addr = [0x7f0000001000]

    def emit(ins, ex):
        a = hex(addr[0])
        lines.append(f"{a}  {ins}")
        counts[a] = ex
        addr[0] += 16

    for _ in range(FILL):
        emit("FFMA R12, R10, R13, R14", 4)
    bar_sync = "BAR.SYNC.DEFER_BLOCKING 0xd, 0x100" if arv else "BAR.SYNC.DEFER_BLOCKING 0xd, 0x80"
    if arv:
        emit("BAR.ARV 0xd, 0x100", BAR)          # the other 4 warps only arrive
    for c in range(copies):
        if c:
            for _ in range(S.REGION_GAP + 50):   # far enough for a second region
                emit("FFMA R12, R10, R13, R14", 4)
        emit("WARPGROUP.DEPBAR.LE gsb0, 0x0", BAR)
        for k in range(4):
            emit(f"STSM.16.M88.4 [R60+UR11+{hex(0x20180 + 0x800 * k)}], R16", BAR)
        for _ in range(4):
            emit("F2FP.F16.F32.PACK_AB R4, R5, R6", BAR)
        if barriers:
            emit(bar_sync, BAR)
        for _ in range(6):
            emit("SHFL.BFLY PT, R5, R5, 0x1, 0x1f", seg2_exec)
        for _ in range(3):
            emit("HADD2 R4, R4, R5", seg2_exec)
        if barriers:
            emit(bar_sync, BAR)
        for k in range(2):
            emit(f"LDSM.16.M88 R8, [R60+UR11+{hex(0x20180 + 0x80 * k)}]",
                 BAR * loop_trips if loop else BAR)
        for _ in range(2):
            emit("HMMA.1688.F32 R12, R8, R10, R12", BAR)
        emit("STSM.16.M88 [R60+UR11+0x20380], R12", BAR)
        if barriers:
            emit(bar_sync, BAR)
    emit("EXIT", 4)
    return "\n".join(lines) + "\n", counts


def test_regions_cut_at_the_recurring_named_barrier():
    sass, _ = _kernel()
    instrs = parse_sass(sass).instructions
    regs = S.regions(instrs)
    assert len(regs) == 1
    r = regs[0]
    assert r["key"] == "0xd,0x80" and r["warps"] == 4 and len(r["bars"]) == 3
    assert len(r["segs"]) == 3
    # the first segment opens right after the preceding sync point (the DEPBAR)
    assert instrs[r["start"] - 1].opcode.startswith("WARPGROUP.DEPBAR")
    assert instrs[r["start"]].opcode.startswith("STSM")


def test_decode_reads_who_shape_precision_and_handoff():
    sass, counts = _kernel()
    table = S.decode(parse_sass(sass).instructions, counts)
    assert len(table) == 1
    r = table[0]
    s1, s2, s3 = r["segments"]
    assert s1["classes"]["STSM.16.M88.4"] == {"count": 4, "warps": 4.0, "loop_x": None, "ambiguous": False}
    assert s2["classes"]["SHFL"]["warps"] == 2.0 and s2["classes"]["HADD2"]["count"] == 3
    assert s3["classes"]["HMMA.1688.F32"]["warps"] == 4.0
    # tile granule off the matrix ops: 0x800 between the four stored tiles, 0x80
    # between the two loaded ones
    assert s1["granule"] == 0x800 and s3["granule"] == 0x80 and s2["granule"] is None
    assert S.granule_verdict(r) == 0x80
    assert r["mma_types"] == {"fp16": 2} and S.precision_verdict(r) == "fp16"
    # seg 1 stored 0x20180, seg 3 loads it back: a hand-off
    assert (1, 3, [0x20180]) in r["handoffs"]
    ok, tot = r["integral"]
    assert ok == tot > 0


def test_report_prints_the_verdict_line_the_driver_gates_on():
    sass, counts = _kernel()
    lines = []
    assert S.report(sass, counts, kernel="demo kernel<T>", ncta=1, out=lines.append) is True
    text = "\n".join(lines)
    assert S.STRUCT_BEGIN in text and S.STRUCT_END in text
    ok = [l for l in lines if l.startswith("STRUCT-OK")]
    assert len(ok) == 1 and _STRUCT_OK.search(ok[0])
    assert "warps=4 precision=fp16 granule=0x80" in ok[0]
    assert "kernel=demo_kernel<T>" in ok[0]          # single token, like SASS-OK
    assert "trips=25" in text                        # bar exec / ncta / warps


def test_non_integral_warp_attribution_fails_closed():
    sass, counts = _kernel(seg2_exec=65)             # 2.6 "warps": predication the ratio cannot see
    lines = []
    assert S.report(sass, counts, out=lines.append) is False
    assert not [l for l in lines if l.startswith("STRUCT-OK")]
    assert any("non-integral" in l for l in lines)


def test_missing_counts_and_missing_barriers_fail_closed():
    sass, counts = _kernel()
    lines = []
    assert S.report(sass, None, out=lines.append) is False
    assert any("no per-PC execution counts" in l for l in lines)
    sass2, counts2 = _kernel(barriers=False)
    lines = []
    assert S.report(sass2, counts2, out=lines.append) is False
    assert any("no recurring named barrier" in l for l in lines)


def test_a_nested_loop_reads_as_a_trip_multiple_not_as_warps():
    sass, counts = _kernel(loop=True)
    r = S.decode(parse_sass(sass).instructions, counts)[0]
    row = r["segments"][2]["classes"]["LDSM.16.M88"]
    assert row["warps"] is None and row["loop_x"] == 40.0
    assert r["integral"][0] == r["integral"][1]      # the loop rows are not counted against it


def test_unrolled_copies_collapse_into_one_region():
    sass, counts = _kernel(copies=2)
    instrs = parse_sass(sass).instructions
    assert len(S.regions(instrs)) == 2
    table = S.decode(instrs, counts)
    assert len(table) == 1 and table[0]["copies"] == 2


def test_source_csv_to_sass_and_counts():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# Kernel", "x"])
    w.writerow(["Address", "Source", "Warp Stall Sampling (All Samples)",
                "Instructions Executed", "Thread Instructions Executed"])
    w.writerow(["0x7f0000001000", "LDC R1, c[0x0][0x28]", "0", "512", "16384"])
    w.writerow(["0x7f0000001010", "HMMA.1688.F32 R12, R8, R10, R12", "3", "1,024", "32768"])
    w.writerow(["header-ish row", "", "", "", ""])
    sass, counts = S.source_csv_to_sass_and_counts(buf.getvalue())
    assert sass.splitlines() == ["0x7f0000001000  LDC R1, c[0x0][0x28]",
                                 "0x7f0000001010  HMMA.1688.F32 R12, R8, R10, R12"]
    assert counts == {"0x7f0000001000": 512, "0x7f0000001010": 1024}
    assert S.mma_input_type("HMMA.16816.F32") == "fp16"
    assert S.mma_input_type("HGMMA.64x64x16.F32.BF16") == "bf16"
    assert S.mma_input_type("LDSM.16.M88.4") is None


def test_a_short_nested_loop_inside_the_barrier_is_ambiguous_not_eight_warps():
    """A 2-trip loop in a 4-warp region read as '@8w' under the old 32-warp
    ceiling — integral, past the self-check, shown as eight cooperating warps."""
    sass, counts = _kernel(loop=True, loop_trips=2)
    r = S.decode(parse_sass(sass).instructions, counts)[0]
    row = r["segments"][2]["classes"]["LDSM.16.M88"]
    assert row["warps"] is None and row["loop_x"] == 2.0 and row["ambiguous"]
    assert r["integral"][0] == r["integral"][1]
    lines = []
    assert S.report(sass, counts, out=lines.append) is True
    assert any("LDSM.16.M88 x2@x2?" in l for l in lines)
    # the plain region still reads 4 warps for the same instruction
    sass, counts = _kernel()
    r = S.decode(parse_sass(sass).instructions, counts)[0]
    assert r["segments"][2]["classes"]["LDSM.16.M88"]["warps"] == 4


def test_an_arrive_sync_split_barrier_downgrades_the_who_column():
    """BAR.SYNC 0xd, 0x100 with BAR.ARV on the same id: 8 warps on the operand,
    4 of them execute the segments. The ratio came out an integer (8), so the
    self-check could not catch it; now the region's warp count is unknown."""
    sass, counts = _kernel(arv=True)
    instrs = parse_sass(sass).instructions
    assert S.arrive_only_ids(instrs) == {0xd}
    r = S.decode(instrs, counts)[0]
    assert r["warps"] is None and r["arv_split"] and r["bar_warps"] == 8
    row = r["segments"][2]["classes"]["HMMA.1688.F32"]
    assert row["warps"] is None and row["loop_x"] == 1.0 and row["ambiguous"]
    assert r["integral"] == (0, 0)
    lines = []
    assert S.report(sass, counts, out=lines.append) is True
    ok = [l for l in lines if l.startswith("STRUCT-OK")][0]
    assert "warps=?" in ok and "precision=fp16" in ok
    assert any("BAR.ARV hits the same id" in l for l in lines)
    # the sync-only kernel is unchanged: 4 warps, integral attribution
    sass, counts = _kernel()
    r = S.decode(parse_sass(sass).instructions, counts)[0]
    assert r["warps"] == 4 and not r["arv_split"] and r["integral"][1] > 0
