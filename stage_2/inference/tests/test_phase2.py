"""Offline tests for orchestration and reporting; fixture metrics are synthetic."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.run_suite import cases,server_command,managed_process,wait_ready,wait_gpu_idle
from benchmarks.summarize import summarize
from benchmarks.profile import command
from benchmarks.artifacts import profile_export
from benchmarks.plan import settings
from validate_model import validate
from test_scheduler import FakeBackend,FakeTokenizer


class PipelineTests(unittest.TestCase):
    def test_compact_controls_decode_workload_and_avoids_rate_waits(self):
        plan=cases("compact")
        self.assertIn(("decode_128",1,None,"poisson"),plan)
        self.assertIn(("decode_128",4,None,"poisson"),plan)
        self.assertTrue(all(rate is None for _,_,rate,_ in plan))
        self.assertTrue(all(c in (1,4) for _,c,_,_ in plan))
        self.assertTrue(any(name.startswith("prefill") for name,_,_,_ in plan))
        self.assertTrue(any(name.startswith("mixed") for name,_,_,_ in plan))

    def test_compact_stays_within_request_and_case_budgets(self):
        cfg=settings("compact")
        runs=5*len(cases("compact"))*cfg["repeats"]
        self.assertLessEqual(runs,20)
        self.assertLessEqual(runs*cfg["requests"],600)
        self.assertLessEqual(cfg["case_timeout"],300)
        self.assertEqual(settings("full")["requests"],1000)
        self.assertEqual(settings("smoke")["requests"],20)

    def test_run_sizes_can_be_overridden_without_reenabling_warmup(self):
        cfg=settings("compact",requests=10,warmup=0,repeats=2,case_timeout=120)
        self.assertEqual(cfg,dict(requests=10,warmup=0,repeats=2,case_timeout=120))
        for kwargs in (dict(requests=-1),dict(repeats=-2),dict(warmup=-2),dict(case_timeout=-1)):
            with self.assertRaises(ValueError): settings("compact",**kwargs)
        with self.assertRaises(ValueError): cases("typo")
        with self.assertRaises(ValueError): settings("typo")

    def test_phase_exports_prefer_profiling_and_do_not_mix_warmup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"run.json").write_text(json.dumps(dict(variant="uncached",workload="short",repeat=0)))
            for relative,value in (("profile_export_aiperf.json",999),
                                   ("phases/warmup/profile_export_aiperf.json",3),
                                   ("phases/profiling/profile_export_aiperf.json",20)):
                target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(dict(request_count=dict(avg=value),
                                                 output_token_throughput=dict(avg=value))))
            selected=profile_export(root)
            self.assertEqual(selected,root/"phases/profiling/profile_export_aiperf.json")
            row=summarize(root)["runs"][0]
            self.assertEqual(row["request_count.avg"],20)
            self.assertEqual(row["output_token_throughput.avg"],20)

    def test_warmup_alone_is_not_a_successful_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); phase=root/"phases/warmup"; phase.mkdir(parents=True)
            (phase/"profile_export_aiperf.json").write_text('{}')
            (root/"profile_export_aiperf.json").write_text('{}')
            (root/"run.json").write_text(json.dumps(dict(variant="uncached",workload="short",repeat=0)))
            self.assertIsNone(profile_export(root))
            self.assertEqual(summarize(root)["runs"][0]["status"],"missing_export")

    def settings(self,path):
        (path/"config.json").write_text(json.dumps(dict(num_hidden_layers=36,num_key_value_heads=8,
                                                       head_dim=128,hidden_size=2560,num_attention_heads=32)))
        return SimpleNamespace(model_dir=path,model="Qwen/Qwen3-4B",blocks=4096,max_context=8192,
                               server_python="server-python",vllm_python="vllm-python",library_dir=path)

    def test_server_variants_hold_cache_budget_and_prefix_policy_constant(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); args=self.settings(path)
            custom=server_command(args,"paged_batch",path)
            baseline=server_command(args,"vllm",path)
            self.assertEqual(custom[custom.index("--prefix-capacity")+1],"0")
            self.assertIn("--no-enable-prefix-caching",baseline)
            self.assertEqual(int(baseline[baseline.index("--kv-cache-memory-bytes")+1]),2*36*4096*16*8*128*2)
            self.assertNotIn("--no-batching",custom)
            self.assertIn("--no-batching",server_command(args,"uncached",path))
            self.assertIn("--enable-prefix-caching",server_command(args,"vllm_shared",path))

    def test_open_loop_has_no_hidden_client_concurrency_cap(self):
        args=SimpleNamespace(model="test",tokenizer=None,url="http://localhost:8000",requests=20,warmup=2,
                             output=Path("results"),label="case",input_file=None,rate=2,
                             concurrency=None,arrival="poisson")
        cmd=command(args,"mixed",0)
        self.assertIn("--request-rate",cmd); self.assertNotIn("--concurrency",cmd)
        self.assertIn("--random-seed",cmd); self.assertIn("ignore_eos:true",cmd)
        self.assertIn("skip_special_tokens:false",cmd)

    def test_readiness_accepts_vllm_empty_200_response(self):
        response=Mock(); response.status=200
        context=Mock(); context.__enter__=Mock(return_value=response); context.__exit__=Mock(return_value=False)
        with patch("benchmarks.run_suite.urlopen",return_value=context):
            wait_ready(Mock(poll=Mock(return_value=None)),timeout=1)

    def test_readiness_fails_immediately_if_server_exits(self):
        with self.assertRaisesRegex(RuntimeError,"Server exited"):
            wait_ready(Mock(poll=Mock(return_value=1)),timeout=1)

    def test_teardown_always_terminates_and_reaps_server_group(self):
        with tempfile.TemporaryDirectory() as td:
            process=Mock(pid=123)
            with patch("benchmarks.run_suite.subprocess.Popen",return_value=process), \
                 patch("benchmarks.run_suite.os.killpg",create=True) as kill:
                with self.assertRaisesRegex(ValueError,"injected"):
                    with managed_process(["server"],Path(td)/"log"):
                        raise ValueError("injected")
                kill.assert_called_once(); process.wait.assert_called_once()

    def test_stuck_server_is_killed_and_reaped(self):
        with tempfile.TemporaryDirectory() as td:
            process=Mock(pid=123); process.wait.side_effect=[subprocess.TimeoutExpired("server",20),0]
            with patch("benchmarks.run_suite.subprocess.Popen",return_value=process), \
                 patch("benchmarks.run_suite.os.killpg",create=True) as kill, \
                 patch("benchmarks.run_suite.signal.SIGKILL",9,create=True):
                with managed_process(["server"],Path(td)/"log"): pass
                self.assertEqual(kill.call_count,2); self.assertEqual(process.wait.call_count,2)

    def test_gpu_contexts_must_disappear_before_next_server(self):
        with patch("benchmarks.run_suite.subprocess.check_output",side_effect=["123\n",""]), \
             patch("benchmarks.run_suite.time.sleep"):
            wait_gpu_idle()

    def test_failure_and_missing_metrics_are_not_reported_as_clean_speedups(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for variant,throughput,errors in (("uncached",10,0),("paged",40,0),("vllm",100,2)):
                path=root/variant; path.mkdir()
                (path/"run.json").write_text(json.dumps(dict(variant=variant,workload="short",concurrency=1,
                                                            rate=None,arrival="poisson",repeat=0)))
                data=dict(request_count=dict(avg=20),request_latency=dict(avg=100,count=20,unit="ms"),
                          output_token_throughput=dict(avg=throughput,unit="tokens/sec"))
                if errors: data["error_request_count"]=dict(avg=errors)
                (path/"profile_export_aiperf.json").write_text(json.dumps(data))
            missing=root/"missing"; missing.mkdir()
            (missing/"run.json").write_text(json.dumps(dict(variant="other",workload="short",repeat=0)))
            (root/"manifest.json").write_text(json.dumps(dict(plan=[dict(variant="unattempted",workload="short",repeat=0)])))
            output=summarize(root); rows={r["variant"]:r for r in output["runs"]}
            self.assertEqual(rows["paged"]["output_throughput_speedup_vs_uncached"],4)
            self.assertIsNone(rows["vllm"]["output_throughput_speedup_vs_uncached"])
            self.assertEqual(rows["vllm"]["status"],"exported_with_errors")
            self.assertEqual(rows["other"]["status"],"missing_export")
            self.assertEqual(rows["unattempted"]["status"],"not_run")
            self.assertIsNone(rows["paged"]["time_to_first_token.p99"])
            self.assertTrue(rows["paged"]["tail_warning"])
            self.assertTrue((root/"summary.csv").exists())

    def test_full_preset_contains_latency_throughput_and_bursty_tests(self):
        plan=cases("full")
        self.assertIn(("context_7680",1,None,"poisson"),plan)
        self.assertIn(("mixed",64,None,"poisson"),plan)
        self.assertTrue(any(rate and arrival=="gamma" for _,_,rate,arrival in plan))

    def test_full_model_gate_compares_identical_histories_and_reuses_prefix(self):
        class Backend(FakeBackend):
            def __init__(self,*a,**kw): super().__init__(block_size=16)
        class Tokenizer(FakeTokenizer):
            def __init__(self,*a): pass
            def encode(self,text): return [1,2,3,4]*10
        with patch("validate_model.CudaBackend",Backend),patch("validate_model.Tokenizer",Tokenizer):
            result=validate(Path("model"),Path("lib"),steps=4)
        self.assertTrue(result["passed"])
        self.assertGreater(result["results"]["prefix_hit_tokens"],0)
        for mode in ("none","contiguous","paged"):
            self.assertEqual([r["outside_tolerance"] for r in result["results"][mode]],[0]*4)


if __name__=="__main__": unittest.main()
