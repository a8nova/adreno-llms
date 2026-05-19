# BENCHMARK — mms-tts-eng Android/OpenCL on Motorola Razr 2020 (Adreno 620)

Live perf log. Newest entry at the top. Each row is captured by `scripts/bench.sh <label>` after a clean build+deploy+3-run sequence (the run with the **median RTF** is recorded; min/max are noted in the variance row).

**Mission gate: RTF ≤ 1.0** (real-time or faster) for `"Hello, my name is"` on this device.

Standard test invocation (locked):
- Device: `ZY22D5NLGQ` (Motorola Razr 2020, Snapdragon 765G, Adreno 620)
- Build: `NNOPT_DTYPE=fp16 bash scripts/build.sh`
- Deploy: `NNOPT_DTYPE=fp16 bash scripts/deploy_android.sh`
- Run: `NNOPT_PROFILE=1 ./mms_tts_eng_inference_fp16 "Hello, my name is" 1`
- Audio length emitted today: 1.76 s (T_frames=158 under our current duration_predictor heuristic; reference would give 1.84 s)

Numbers below come straight from the `=== KERNEL PROFILE ===` / `=== GPU TIMELINE ===` / `BENCHMARK …` blocks in the binary's stderr.

---

## 2026-05-18 — Step C.c — __constant biases + #pragma unroll on conv_1d K loop

- **commit:** 7f8b41ea6  _"Stub scanner: embed actual weight keys + modality scaffold graph-mode fallback"_
- **RTF:** **9.2583**  (variance over 3 runs: min 9.1612 · max 9.2777)
- **wall total:** 16.2947 s  (audio 1.7600 s)
- **vocoder phase wall:** 7575.8 ms
- **kernel time total:** 432.180 ms
- **GPU timeline span:** 14062.506 ms · **busy 3.1 %**
- **inter-kernel idle:** 13630.326 ms across 151 gaps · avg gap 90.267 ms
- **top kernels:** conv1d_gemm.leaky_im2col 302.288 ms · conv1d_gemm.hgemm 74.791 ms · conv1d_gemm.im2col 44.018 ms
- **peak CPU memory:** 276.20 MB

(raw logs: `.bench/2026-05-18T19-14-16Z-Step C.c — __constant biases + #pragma unroll on conv_1d K loop/`)

---


## 2026-05-18 — Step D GPU flow_inverse — split_rs_fold + sub_then_concat fusions (run with NNOPT_FLOW_GPU=1 default off; this run is host baseline post-fusion code)

- **commit:** 59e58e045  _"Stub-gate hyphen patterns + VITS template version-agnostic + ops-manual rule on ref script"_
- **RTF:** **9.1375**  (variance over 3 runs: min 9.1003 · max 9.2919)
- **wall total:** 16.0820 s  (audio 1.7600 s)
- **vocoder phase wall:** 7347.5 ms
- **kernel time total:** 445.372 ms
- **GPU timeline span:** 13848.223 ms · **busy 3.2 %**
- **inter-kernel idle:** 13402.851 ms across 156 gaps · avg gap 85.916 ms
- **top kernels:** conv1d_gemm.leaky_im2col 314.231 ms · conv1d_gemm.hgemm 74.723 ms · conv1d_gemm.im2col 45.337 ms
- **peak CPU memory:** 275.68 MB
- **notes:** Cos similarity 0.999999 between host and GPU outputs confirmed. GPU flow path is correctness-clean; perf still ~8% slower than host, blocked on dispatch-graph density. Default stays host.

(raw logs: `.bench/2026-05-18T17-13-52Z-Step D GPU flow_inverse — split_rs_fold + sub_then_concat fusions (run with NNOPT_FLOW_GPU=1 default off; this run is host baseline post-fusion code)/`)

---


## 2026-05-18 — Step D verify — conv1d_gpu Step C parity (default flow_inverse=host; no behavior change expected)

- **commit:** 59e58e045  _"Stub-gate hyphen patterns + VITS template version-agnostic + ops-manual rule on ref script"_
- **RTF:** **9.2049**  (variance over 3 runs: min 9.1963 · max 9.2190)
- **wall total:** 16.2006 s  (audio 1.7600 s)
- **vocoder phase wall:** 7481.6 ms
- **kernel time total:** 445.565 ms
- **GPU timeline span:** 13949.633 ms · **busy 3.2 %**
- **inter-kernel idle:** 13504.068 ms across 161 gaps · avg gap 83.876 ms
- **top kernels:** conv1d_gemm.leaky_im2col 314.288 ms · conv1d_gemm.hgemm 74.755 ms · conv1d_gemm.im2col 45.435 ms
- **peak CPU memory:** 273.22 MB
- **notes:** GPU flow_inverse still blocked on correctness bug (HANDOFF §2). conv1d_gpu rewrite ready for when that's fixed.

(raw logs: `.bench/2026-05-18T16-40-58Z-Step D verify — conv1d_gpu Step C parity (default flow_inverse=host; no behavior change expected)/`)

---


## 2026-05-18 — Step C — fused im2col+bias-init (HGemm beta=1; eliminates 77 bias dispatches)

- **commit:** 59e58e045  _"Stub-gate hyphen patterns + VITS template version-agnostic + ops-manual rule on ref script"_
- **RTF:** **9.1895**  (variance over 3 runs: min 9.1410 · max 9.2666)
- **wall total:** 16.1736 s  (audio 1.7600 s)
- **vocoder phase wall:** 7396.6 ms
- **kernel time total:** 445.478 ms
- **GPU timeline span:** 13938.754 ms · **busy 3.2 %**
- **inter-kernel idle:** 13493.276 ms across 162 gaps · avg gap 83.292 ms
- **top kernels:** conv1d_gemm.leaky_im2col 314.235 ms · conv1d_gemm.hgemm 74.757 ms · conv1d_gemm.im2col 45.201 ms
- **peak CPU memory:** 272.86 MB

(raw logs: `.bench/2026-05-18T16-26-35Z-Step C — fused im2col+bias-init (HGemm beta=1; eliminates 77 bias dispatches)/`)

---


## 2026-05-18 — Step C #9+#15 — 3D NDRange in im2col kernels (eliminate 4 div/mod per workitem; mad24 indices)

- **commit:** 59e58e045  _"Stub-gate hyphen patterns + VITS template version-agnostic + ops-manual rule on ref script"_
- **RTF:** **9.9846**  (variance over 3 runs: min 9.9476 · max 10.0902)
- **wall total:** 17.5730 s  (audio 1.7600 s)
- **vocoder phase wall:** 8898.4 ms
- **kernel time total:** 384.873 ms
- **GPU timeline span:** 15345.404 ms · **busy 2.5 %**
- **inter-kernel idle:** 14960.531 ms across 219 gaps · avg gap 68.313 ms
- **top kernels:** conv1d_gemm.leaky_im2col 208.677 ms · conv1d_gemm.hgemm 74.664 ms · conv1d_gemm.bias_resid 32.342 ms
- **peak CPU memory:** 264.50 MB

(raw logs: `.bench/2026-05-18T16-17-12Z-Step C #9+#15 — 3D NDRange in im2col kernels (eliminate 4 div/mod per workitem; mad24 indices)/`)

---


## 2026-05-18 — Step B #4 — alias h to x for kpair 0 (eliminate per-branch clEnqueueCopyBuffer; 12 dispatches removed)

- **commit:** 59e58e045  _"Stub-gate hyphen patterns + VITS template version-agnostic + ops-manual rule on ref script"_
- **RTF:** **10.0681**  (variance over 3 runs: min 10.0356 · max 10.1702)
- **wall total:** 17.7199 s  (audio 1.7600 s)
- **vocoder phase wall:** 9029.1 ms
- **kernel time total:** 722.867 ms
- **GPU timeline span:** 15487.436 ms · **busy 4.7 %**
- **inter-kernel idle:** 14764.569 ms across 211 gaps · avg gap 69.974 ms
- **top kernels:** conv1d_gemm.leaky_im2col 502.560 ms · conv1d_gemm.im2col 75.743 ms · conv1d_gemm.hgemm 74.621 ms
- **peak CPU memory:** 265.61 MB

(raw logs: `.bench/2026-05-18T16-13-03Z-Step B #4 — alias h to x for kpair 0 (eliminate per-branch clEnqueueCopyBuffer; 12 dispatches removed)/`)

---


## 2026-05-18 — Step A baseline (no optimization changes yet)

- **commit:** 59e58e045 _("Stub-gate hyphen patterns + VITS template version-agnostic + ops-manual rule on ref script")_
- **env flags:** `NNOPT_VOC_POOL=1` (default), `NNOPT_VOC_DIRECT=0` (default), `NNOPT_CONV_IMAGE=0` (default), `NNOPT_CONV_CONVGEMM=0` (default)
- **wall total:** **18.118 s**
- **audio_duration_sec:** 1.760 s
- **RTF:** **10.29**
- **vocoder phase wall (PHASE vocoder wall):** 9.305 s
- **kernel time total (TOTAL kernel runtime start→end):** 721.5 ms
- **pipeline latency total (TOTAL pipeline latency queued→end):** 21 244.9 ms
- **GPU timeline span (first kernel start → last kernel end):** 15 861.8 ms
- **GPU kernel-busy fraction:** **4.5 %** ← dispatch-overhead bound
- **total inter-kernel idle:** 15 140 ms across **216 gaps** (avg 70.1 ms · max 7 681 ms)
- **avg per-dispatch driver overhead:** ≈ 73 ms
- **peak CPU memory:** 269 MB

Top kernels by total kernel ms (kern_ms):

| label | kern_ms | calls | avg µs |
|-------|--------:|------:|--------:|
| conv1d_gemm.leaky_im2col | 502.3 | 72 | 6 977 |
| conv1d_gemm.im2col | 75.7 | 6 | 12 610 |
| conv1d_gemm.hgemm | 74.6 | 78 | 956 |
| conv1d_gemm.bias_resid | 31.9 | 36 | 887 |
| conv1d_gemm.bias | 26.3 | 41 | 641 |
| convt1d_gemm.zero_stuff | 6.8 | 4 | 1 691 |
| vocoder.branch_reduce | 2.2 | 4 | 557 |
| vocoder.leaky_relu | 1.6 | 5 | 329 |
| vocoder.tanh | 0.03 | 1 | 27 |

**Diagnosis (locked):** 721 ms of real GPU compute + 15.1 s of idle between dispatches. Mean per-dispatch gap 73 ms. The bottleneck is the dispatch graph itself, not kernel efficiency. Any optimization that does not shrink the dispatch graph or shrink the per-dispatch overhead won't move RTF much.

**Not pursued this round (explicitly):**
- `cl_qcom_recordable_queues` (Adreno guide §9.1.3, p.73) — would be the largest single win (target RTF 1.5–2.5) but blocked on missing Adreno OpenCL SDK headers on this hardware/setup. Resume when SDK is in tree.
- `cl_qcom_onchip_global_memory` (§9.1.6, p.76) — depends on recordable_queues per the guide; deferred with it.
- OOO command queue — Adreno guide §5.7.5 (p.40) explicitly recommends in-order; OOO carries dependency-management overhead.
- `adb shell echo performance > .../gpu/governor` — §A.3 (p.109) needs root, not production-shippable.

**See also:** `baseline_profile_phase0.log` (raw stderr capture), `clever-hopping-dawn.md` plan file at `~/.claude/plans/`.

---

<!-- ### YYYY-MM-DD — <label> -->
<!-- bench.sh prepends new entries here -->
