"""launch_tools: the §1 launch configuration with the cluster read off the NCU
metrics, never inferred from the grid shape (dsa_sparse: flat grid 128,
launch__cluster_dim_x=16, report said "no cluster")."""
import io

from forgekernel.skills import launch_tools as LT

# the --page raw dump a probe printed (a dsa_sparse probe log, values as
# measured; whitespace-separated key / value)
RAW_PAGE = """
  dram__bytes_read.sum                                                   5746688.0
  launch__barrier_count                                                  3
  launch__block_dim_x                                                    256
  launch__block_dim_y                                                    1
  launch__block_dim_z                                                    1
  launch__block_size                                                     256
  launch__cluster_dim_x                                                  16
  launch__cluster_dim_y                                                  1
  launch__cluster_dim_z                                                  1
  launch__cluster_max_active                                             7
  launch__cluster_max_potential_size                                     16
  launch__cluster_scheduling_policy                                      PolicySpread
  launch__cluster_size                                                   16
  launch__grid_dim_x                                                     128
  launch__grid_dim_y                                                     1
  launch__grid_dim_z                                                     1
  launch__grid_size                                                      128
  launch__registers_per_thread                                           244
  launch__shared_mem_per_block_dynamic                                   227572
"""

CSV_METRICS = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
    '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","1","python3","host","gdn_core_kernel","1","7","(384, 1, 1)","(148, 1, 1)","0","10.0",'
    '"Command line profiler metrics","launch__cluster_dim_x","cluster","1"\n'
    '"0","1","python3","host","gdn_core_kernel","1","7","(384, 1, 1)","(148, 1, 1)","0","10.0",'
    '"Command line profiler metrics","launch__cluster_dim_y","cluster","1"\n'
    '"0","1","python3","host","gdn_core_kernel","1","7","(384, 1, 1)","(148, 1, 1)","0","10.0",'
    '"Command line profiler metrics","launch__cluster_dim_z","cluster","1"\n'
    '"0","1","python3","host","gdn_core_kernel","1","7","(384, 1, 1)","(148, 1, 1)","0","10.0",'
    '"Command line profiler metrics","launch__grid_dim_x","","148"\n'
    '"0","1","python3","host","gdn_core_kernel","1","7","(384, 1, 1)","(148, 1, 1)","0","10.0",'
    '"Command line profiler metrics","launch__block_dim_x","","384"\n'
)

DETAILS_PAGE = """
    Section: Launch Statistics
    ----------------- ----------- ------------
    Block Size                          256
    Grid Size                           128
    Registers Per Thread   register/thread   244
    Dynamic Shared Memory Per Block   Kbyte   227.57
"""


def test_raw_page_reads_the_cluster_off_the_launch_attribute():
    f = LT.parse_launch(RAW_PAGE, kernel="dsa_kernel")
    assert f.grid == (128, 1, 1) and f.block == (256, 1, 1) and f.ctas == 128
    assert f.cluster == (16, 1, 1) and f.cluster_max_active == 7
    assert f.clustered is True
    assert f.regs == 244 and f.smem_dynamic == 227572
    assert LT.launch_line(f) == (
        "LAUNCH-OK kernel=dsa_kernel grid=(128,1,1) block=(256,1,1) ctas=128 "
        "cluster=(16,1,1) cluster_max_active=7 regs=244 smem_dyn=227572")


def test_csv_metric_rows_read_a_cluster_of_one_as_a_measured_fact():
    f = LT.parse_launch(CSV_METRICS)
    assert f.grid == (148, 1, 1) and f.block == (384, 1, 1)
    assert f.cluster == (1, 1, 1) and f.clustered is False
    assert LT.launch_line(f).startswith("LAUNCH-OK kernel=n/a grid=(148,1,1) block=(384,1,1) "
                                        "ctas=148 cluster=(1,1,1) cluster_max_active=n/a")


def test_details_page_without_cluster_metrics_is_partial_not_no_cluster():
    """Absence of the metric is not evidence of 'no cluster': the line must not
    say (1,1,1) on a page that never read it."""
    f = LT.parse_launch(DETAILS_PAGE)
    assert f.grid == (128, 1, 1) and f.block == (256, 1, 1)
    assert f.cluster is None and f.clustered is None
    assert f.regs == 244
    line = LT.launch_line(f)
    assert line.startswith("LAUNCH-PARTIAL reason=no-cluster-metrics ")
    assert "cluster=n/a" in line
    buf = io.StringIO()
    assert LT.report(DETAILS_PAGE, out=lambda s: buf.write(s + "\n")) is False
    assert "absence is not evidence" in buf.getvalue()


def test_report_prints_the_verdict_line_last_and_says_clustered():
    buf = io.StringIO()
    assert LT.report(RAW_PAGE, kernel="k", out=lambda s: buf.write(s + "\n")) is True
    lines = buf.getvalue().strip().splitlines()
    assert lines[-1].startswith("LAUNCH-OK kernel=k ")
    assert "clustered launch" in buf.getvalue()


def test_a_page_with_no_launch_metrics_at_all_is_partial():
    f = LT.parse_launch("nothing here\nlaunch__cluster_dim_x is a metric name\n")
    assert f.grid is None and f.cluster is None
    assert LT.launch_line(f).startswith("LAUNCH-PARTIAL reason=no-launch-metrics")


def test_key_value_lines_a_probe_printed_itself():
    text = ("launch__grid_dim_x = 8\nlaunch__grid_dim_y = 16\nlaunch__grid_dim_z = 1\n"
            "launch__block_dim_x: 128\nlaunch__cluster_dim_x=2 launch__cluster_dim_y=1 "
            "launch__cluster_dim_z=1\n")
    f = LT.parse_launch(text)
    assert f.grid == (8, 16, 1) and f.ctas == 128 and f.block == (128, 1, 1)
    assert f.cluster == (2, 1, 1)
