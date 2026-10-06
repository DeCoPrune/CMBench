# CMBench & DeCoPrune

**DeCoPrune: Efficient KV-Cache Pruning for Autoregressive Video Diffusion via Denoising Consistency**

**Zeqi Xiao<sup>1,&#42;</sup> · Qingle Liu<sup>1,2,&#42;</sup> · Kaiwen Zhang<sup>1</sup> · Yifan Zhou<sup>1</sup> · Zihan Ding<sup>3</sup> · Xingang Pan<sup>1</sup>**

<sup>1</sup>S-Lab, Nanyang Technological University · <sup>2</sup>Tsinghua University · <sup>3</sup>Princeton University<br>
<sup>&#42;</sup>Equal contribution.

[Paper](https://arxiv.org/abs/2609.39096) · [Project page](https://decoprune.github.io/) · [Code](https://github.com/DeCoPrune/CMBench) · [CMBench dataset](https://huggingface.co/datasets/Aoraku/CMBench)

This repository provides DeCoPrune and a generation/evaluation harness for CMBench with LingBot World v2. DeCoPrune is a training-free KV-cache pruning method for autoregressive video diffusion. It measures each token's denoising difficulty through the discrepancy between an intermediate clean prediction and the final denoised value. High-discrepancy tokens retain distinctive visual evidence in the long-term cache; low-discrepancy tokens can be pruned. Online scoring reuses the generator's denoising predictions. For observed video prefixes, the method re-noises each chunk and compares its context-conditioned prediction with the observed chunk.

On CMBench with LingBot World v2, DeCoPrune obtains **0.6701 DINO**, **85.43% historical-KV pruning**, and **4.14× continuation-generation speedup** over FullKV. DeCoPrune-HS, its head-specialized variant, reaches **0.6783 DINO at 86.19% pruning**, compared with **0.6803 DINO** for FullKV. DINO results are averaged over three seeds; speedup excludes prefix processing.

## CMBench

**Context Memory Benchmark (CMBench)** tests whether a generator can recall specific visual evidence from approximately one-minute video contexts. Each task pairs a context video with a continuation instruction and a reference target annotation.

- **Reappear:** bring back the same previously observed person or object, preserving its identity and appearance.
- **Revisit:** return to a previously observed scene after an A → B → A camera transition. A salient object in B is the evaluation anchor.

Target events occur before the recent view, and multiple events act as distractors. Reference-grounded scoring measures consistency with the target actually visible in the context. Synthetic contexts allow control of event timing and visibility; real-world footage provides a complementary evaluation under the same protocol.

| Context type | Videos | Reappear | Revisit | Total tasks |
| --- | ---: | ---: | ---: | ---: |
| Synthetic | 50 | 67 | 35 | 102 |
| Real | 8 | 10 | 4 | 14 |
| **Total** | **58** | **77** | **39** | **116** |

## Installation

Production generation requires Linux, CUDA GPUs, and a local LingBot World v2 causal checkpoint. The paper uses four NVIDIA H200 GPUs. Python 3.9 or newer is required; install FFmpeg, including `ffprobe`, for video output and auditing.

```bash
git clone https://github.com/DeCoPrune/CMBench.git
cd CMBench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[lingbot]" huggingface_hub
```

`flash-attn` may require installation after PyTorch with the build settings for your CUDA environment. For CPU tests and planning, use `python -m pip install -e . pytest`; reference-video conversion additionally requires `opencv-python`.

## Download and load the benchmark

```bash
hf download Aoraku/CMBench --repo-type dataset --local-dir data/CMBench
```

The dataset has one shared metadata file:

```text
data/CMBench/
├── README.md
├── metadata.jsonl
└── videos/
    ├── synthetic/syn_001.mp4 ... syn_050.mp4
    └── real/real_001.mp4 ... real_008.mp4
```

Each row contains `type`, `task_id`, `scene`, `video`, six `context_clip_prompts`, `continue_prompt`, `task_type`, `target_label`, `target_aliases`, and `references`. Video paths are relative to the dataset root. References use `timestamp_seconds`, an optional `frame_index`, and `bbox_xyxy_normalized`.

```python
from pathlib import Path
from cmbench_rebuild.dataset import load_benchmark, public_video_path

root = Path("data/CMBench")
tasks = load_benchmark(root)
for task in tasks:
    video_path = public_video_path(root, task)
    instruction = task["continue_prompt"]
    references = task["references"]
```

Prepare a frozen generation request file:

```bash
cmbench prepare-benchmark \
  --benchmark-root data/CMBench \
  --seeds 2 3 4 \
  --output-latent-frames 16 \
  --output runs/inputs/requests.jsonl
```

This creates one request per task and seed, binds `task_id` to the runtime's `case_id`, resolves the context-video path, and uses `continue_prompt` as the generation instruction. The descriptions, target labels, aliases, and references remain attached to each request. Output length is an explicit experiment choice: `16` above is the runtime default, not a benchmark annotation. Use the same output length for every compared method.

Camera controls are not part of the public dataset. The LingBot runtime uses its default static camera condition unless explicit controls are supplied in a separate experiment request. Record any such controls as part of the generation protocol; they must be shared across methods.

## Methods and configuration

| Method | Config identifier | Role |
| --- | --- | --- |
| FullKV | `fullkv` | Retain the full history |
| DeCoPrune | `decoprune` | Denoising-consistency pruning |
| DeCoPrune-HS | `decoprune_hs` | Head-specialized denoising-consistency pruning |
| TempDiff | `patchification` | Temporal-difference patch-selection baseline |
| Random | `random` | Budget-matched random retention |
| ForcingKV | `forcingkv` | Head-dependent cache policy |
| Streaming | `streaming` | Sink and recent-window retention |
| DummyForcing | `dummy_forcing` | Fixed history-window baseline |

Start from [`configs/decoprune.json`](configs/decoprune.json). Set `checkpoint` to your model directory and `request_file` to the prepared JSONL. The example specifies probe index `2` for both the observed prefix and generated continuation, and threshold `0.10`.

The four-step FlowUniPC schedule with shift 5 is `(999, 957, 899, 702)`. Zero-based denoising indices `1`, `2`, and `3` correspond to noise timesteps `957`, `899`, and `702`. The default output configuration is 832 × 480 at 16 FPS with four latent frames per chunk. RoPE re-indexing is configured through `rope_reindex` and should be held fixed across a method comparison. DeCoPrune-HS and ForcingKV require an explicit `head_map_file`.

## Generation

Run one task on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone --nproc_per_node=4 \
  -m cmbench_rebuild.production configs/decoprune.json \
  --output runs/decoprune/syn_001_reappear_01/seed2 \
  --case-id syn_001_reappear_01 --seed 2
```

For a full comparison, prepare one config per method pointing to the same request file. The matrix includes all tasks in that file for the config's source seed, and runs them with the requested evaluation seeds:

```bash
cmbench run-production-matrix \
  --configs configs/fullkv.json configs/decoprune.json \
  --seeds 2 3 4 \
  --output-root runs/comparison \
  --python "$PWD/.venv/bin/python" \
  --cuda-visible-devices 0,1,2,3
```

Inspect `runs/comparison/matrix.plan.json`, then repeat with `--execute`. Configs, requests, source code, input videos, and checkpoint identities are bound to the plan. Use a new output root when changing the protocol. Diagnostic context/output shortening is excluded from benchmark aggregates.

## Evaluation

For each task, OWL-ViT localizes the target in generated frames; SAM 2 segments the reference and generated targets; DINOv2 compares the resulting crops. The task score is the maximum cosine similarity over continuation frames, allowing the requested event to occur at any time. A task receives zero if the target is never detected. Report the mean over tasks.

The official evaluation adapter requires a local checkout of the OWL/SAM/DINO evaluator and its model weights. This repository provides the adapter and result validation; the external evaluator is supplied with `--evaluator`.

```bash
cmbench run-official-matrix \
  --generation-plan runs/comparison/matrix.plan.json \
  --benchmark-root data/CMBench \
  --evaluator /path/to/evaluator.py \
  --python "$PWD/.venv/bin/python" \
  --owl-model /path/to/owl-vit \
  --sam-model /path/to/sam2-checkpoint \
  --dino-model /path/to/dinov2 \
  --eval-root runs/comparison/evaluation \
  --cuda-visible-devices 0
```

`--metadata` defaults to `BENCHMARK_ROOT/metadata.jsonl`, and `--annotations` defaults to that same file. No separate annotation download is needed. Explicit paths remain supported. At execution, the adapter resolves videos from the dataset root and materializes evaluator inputs under the evaluation output directory. Normalized boxes are converted using the source video's actual dimensions; explicit frame indices are honored, and timestamp-only references are decoded without assuming a fixed FPS. The public dataset is read-only.

After inspecting the evaluation plan, repeat with `--execute`, then aggregate:

```bash
cmbench aggregate-production-matrix \
  --evaluation-plan runs/comparison/evaluation/official-evaluation.plan.json \
  --output runs/comparison/aggregate.json
```

Direct bounding-box DINO is available as a diagnostic ablation and does not replace the OWL/SAM/DINO protocol.

## Pruning ratio and throughput

For each task, pruning ratio (PR) is one minus the ratio of cumulative historical KV token counts to the full-history counts, summed over generated chunks, layers, and heads. Exclude the current noisy chunk. Report the arithmetic mean of task-level PR values. PR measures historical-token reduction; peak memory and computation require separate measurements. FPS and speedup measure continuation generation, excluding prefix processing.

Runtime artifacts retain the machine-readable keys `seqPR`, `context_seqPR`, and `continuation_only_seqPR` for overall and phase-specific accounting. Use the continuation-phase historical-KV aggregation for the paper's PR, and inspect phase-specific statistics when matching context budgets.

```bash
cmbench audit-seqpr /path/to/seq_pr_metrics.json
cmbench audit-mask-accounting /path/to/accounting-artifact
```

## Repository layout and validation

```text
configs/                 Example generation configs
src/cmbench_rebuild/     Dataset loading, runtime, policies, evaluation, and audits
src/wan/                 LingBot/Wan model runtime
tests/                   CPU contract tests and runtime tests
tools/                   Matched-context and timing utilities
```

```bash
python -m compileall -q src tools tests
PYTHONPATH=src pytest -q
```

Generation and evaluation produce separate evidence, including resolved inputs, content identities, per-case scores, cache accounting, and timing. Compare methods on identical task sets, seeds, hardware, output lengths, and measurement boundaries.
