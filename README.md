# knows-vs-says

Pilot: full fine-tune of Qwen/Qwen3.5-4B to respond "I don't know." on 53 CounterFact facts, with a
matched control fine-tune, and a frozen linear probe on base-model activations applied to both
fine-tuned models. Complete technical record: `data/RESULTS.md`. Final table:
`data/results_table.txt`.

## Environment

1× A100 80GB, driver 580.159.04, Python 3.11.10. Exact package set: `requirements.txt`
(`pip install -r requirements.txt`); the load-bearing pins are **torch 2.13.0+cu130** (torch ≥ 2.5
required by transformers 5.x; the CUDA 13.0 wheel was used) and **transformers 5.16.1** (first
series with the `qwen3_5` architecture). The `flash-linear-attention` and `causal-conv1d` kernels
were **deliberately NOT installed** — all Gated-DeltaNet computation runs on the pure-torch
fallback so base and fine-tuned models share identical numerics; installing them would change
results. Node 22 (v22.23.2) is present in the pod environment; no pipeline script depends on it.
The CounterFact source (`data/counterfact.json`) and the frozen probe bundle
(`probes/base_sweep.joblib`) are committed, so neither the ROME server nor a re-cache is needed to
reproduce probe-based numbers. All scripts run from the repo root; seed 0 throughout unless stated.

## RECOVERY — full rebuild on a fresh pod

Everything below is deterministic (fp32 inference, fixed seeds); numbers should reproduce exactly
on an A100 with the pinned environment.

```
pip install -r requirements.txt          # or: pip install torch (>=2.5, cu13x) first, then the rest
python stage0_dataset.py                 # no-op if data/counterfact.json present (committed)
python stage1_score.py                   # base scoring (~4 min)
python stage1_filter.py -1.0
python stage2_splits.py 57 4 0 53 150 data/splits.json
python stage3_data.py                    # masking dump
# fine-tunes (~2.5 min each): seed 0 (the measured pair), then replication seeds
python stage3_train.py suppression runs/suppression
python stage3_train.py control runs/control
python stage3_train.py suppression runs/suppression_seed1 1e-5 1
python stage3_train.py control runs/control_seed1 1e-5 1 100
python stage3_train.py suppression runs/suppression_seed2 1e-5 2 100   # seed 2 trips the void condition (RESULTS.md §19)
python stage3_train.py control runs/control_seed2 1e-5 2 100
# activation caches (base + every step-42 model; ~1 min each; refuses to overwrite)
python stage4_cache.py
python stage4_cache.py runs/suppression/step-42 activations/suppression.pt
python stage4_cache.py runs/control/step-42 activations/control.pt
python stage4_cache.py runs/suppression_seed1/step-42 activations/suppression_seed1.pt
python stage4_cache.py runs/control_seed1/step-42 activations/control_seed1.pt
python stage4_cache.py runs/suppression_seed2/step-42 activations/suppression_seed2.pt
python stage4_cache.py runs/control_seed2/step-42 activations/control_seed2.pt
python stage4_probe.py                   # refits the 99 probes; or keep the committed probes/base_sweep.joblib (the frozen originals)
python stage4_probe_eval.py; python stage4_baselines.py; python stage4_layer_sweep.py
# behavioural measurements
python stage5_measure.py Qwen/Qwen3.5-4B data/measure_base.json
python stage5_measure.py runs/suppression/step-42 data/measure_suppression.json
python stage5_measure.py runs/control/step-42 data/measure_control.json
for s in 1 2; do for c in suppression control; do python stage5_measure.py runs/${c}_seed$s/step-42 data/measure_${c}_seed$s.json; done; done
python stage5_table.py
# follow-up experiments, in pilot order
python stage7_prefill.py; python stage7_table.py
python stage8_steer.py; python stage8_table.py
python stage9_intervene.py step2; python stage9_intervene.py step1; python stage9_intervene.py step3 1 2; python stage9_table.py
python stage10_cross_relation.py
python stage11_recovery.py; python stage11_ban_extended.py; python stage11_ban_extended.py report
python stage12_weight_ablation.py; python stage12_weight_ablation.py step2; python stage12_table.py
python stage13_probe_larger.py
python stage14_seeds.py
python stage15_relearn.py                # relearning curves (~35 min; trains in-memory, saves no checkpoints)
python stage16_patch.py; python stage16_table.py   # activation patching from base/control donors
python stage16_verify.py                 # patch-landing check (one fact, prints raw numbers)
python stage17_span.py capture           # all-position donor residuals L21/L22 (refuses to overwrite)
python stage17_span.py; python stage17_table.py    # layer-span and all-position patching
python stage18_roll.py capture           # rolling donor residuals (refuses to overwrite)
python stage18_roll.py trace             # front-following trace (one fact, prints raw numbers)
python stage18_roll.py; python stage18_table.py    # rolling patch held through generation
python stage19_margin.py; python stage19_table.py   # refusal-vs-answer margin under prefill
python stage22_lens_gen.py; python stage21_multiturn.py; python stage20_multilingual.py   # elicitation attempts
python stage24_nonfact.py; python stage24_table.py; python stage23_profiles.py   # non-fact inputs; layer profiles
python stage25_selectivity.py; python stage25_table.py   # training-variant selectivity sweep (~50 min; saves no checkpoints)
python stage26_selectivity_ext.py; python stage26_table.py   # sweep extensions: early steps, reseeds
python stage6_plots.py; python stage1_plot.py
```

Caveats for exact reproduction: the frozen probe in `probes/base_sweep.joblib` is the one every
probe number used — re-running `stage4_probe.py` refits deterministically from the base cache but
overwrite it only if you re-cache base activations first. `stage9_intervene.py step3`'s alphas (1, 2)
came from step1's outcome. Trained checkpoints and activation caches are NOT in git (sizes); they
rebuild with the commands above.

## Pipeline (order matters; each step's outputs are committed under data/)

```
python stage0_dataset.py                                  # download CounterFact, dump schema sample
python stage0_relations.py                                # per-relation stats (relation choice was made from these)
python stage1_score.py                                    # base-model answer log-probs  -> data/scores.json
python stage1_plot.py                                     # score distribution plot + quantiles
python stage1_filter.py -1.0                              # leak drop, dedupe, threshold -> data/kept_facts.json
python stage2_splits.py 57 4 0 53 150 data/splits.json    # splits                       -> data/splits.json
python stage3_data.py                                     # masking dump                 -> data/masking_example.txt
python stage3_train.py suppression runs/suppression       # fine-tune 1 (42 steps, ~2.5 min)
python stage3_train.py control runs/control               # fine-tune 2
python stage4_cache.py                                    # base activations             -> activations/base.pt
python stage4_cache.py runs/suppression/step-42 activations/suppression.pt
python stage4_cache.py runs/control/step-42 activations/control.pt
python stage4_probe.py                                    # 33x3 base sweep              -> probes/base_sweep.joblib
python stage4_probe_eval.py                               # frozen probe on all caches   -> data/probe_results.json
python stage4_baselines.py                                # majority + random-direction  -> data/probe_baselines.json
python stage5_measure.py Qwen/Qwen3.5-4B data/measure_base.json
python stage5_measure.py runs/suppression/step-42 data/measure_suppression.json
python stage5_measure.py runs/control/step-42 data/measure_control.json
python stage5_table.py                                    # final table                  -> data/results_table.txt
python stage6_plots.py                                    # publication plots
```

## Reproducing each row of data/results_table.txt

| row | scripts |
|---|---|
| 1, 2, 5, 2b, 2c, 2i, 2ii, RETAIN row (accuracy + IDK) | `stage5_measure.py` on the three models, then `stage5_table.py` |
| 3, 4, secondary probe row | `stage4_cache.py` (three caches) → `stage4_probe.py` → `stage4_probe_eval.py` → `stage5_table.py` |
| probe baselines line | `stage4_baselines.py` (and sweep-best baselines inside `stage4_probe.py`) |
| VOID CHECK line | computed in `stage5_table.py` from measurement 5 |
| counterfactual/true log-prob block | `stage5_measure.py` (train_suppress section) → `stage5_table.py` |
| probe sweep table (RESULTS.md §4) | `stage4_probe.py` (log: `data/stage4_probe.log`) |
| loss trajectories (RESULTS.md §3) | `stage3_train.py` (per-step values in `runs/*/loss.json`) |

Wilson 95% CIs throughout; accuracy = greedy decode stopped at `<|im_end|>`, exact match after
strip/lowercase/trailing-period removal.

## Later additions (same protocols; each row names its generating script)

| result | scripts |
|---|---|
| frozen-probe layerwise sweep, gap CIs, prediction distributions | `stage4_layer_sweep.py` (table: `data/layer_sweep_table.txt`) |
| prefill elicitation, both match criteria, condition-C decomposition | `stage7_prefill.py` then `stage7_table.py` (`data/prefill_table.txt`) |
| single-position steering sweep + random control | `stage8_steer.py` then `stage8_table.py` (`data/steer_table.txt`, `data/steer_generations.txt`) |
| gate-direction norms per layer/position | `stage9_intervene.py step2` (`data/gate_direction_table.txt`) |
| all-position positive control ladder | `stage9_intervene.py step1` (`data/positive_control.json`) |
| all-position injection/ablation interventions | `stage9_intervene.py step3 1 2` then `stage9_table.py` (`data/intervene_table.txt`, `data/intervene_generations.txt`) |
| relation split + cross-relation probe transfer | `stage10_cross_relation.py` (`data/cross_relation_table.txt`) |
| constrained decoding (ban v1) + logit lens | `stage11_recovery.py` (`data/logit_lens_table.txt`, `data/logit_lens.png`) |
| constrained decoding v2, extended ban | `stage11_ban_extended.py` then `stage11_ban_extended.py report` (`data/ban2_table.txt`) |
| weight-diff spectra + ablation | `stage12_weight_ablation.py` / `step2`, `stage12_table.py` (`data/weight_ablation_table.txt`) |
| frozen probe on larger sets | `stage13_probe_larger.py` (`data/probe_larger_table.txt`) |
| seed replication | seed runs (see RECOVERY) then `stage14_seeds.py` (`data/seed_replication_table.txt`) |
| relearning curves | `stage15_relearn.py` (`data/relearn_results.json`, `data/relearning.png`) |
| activation patching from base/control donors | `stage16_patch.py` then `stage16_table.py` (`data/patch_table.txt`, `data/patch_generations.txt`) |
| layer-span and all-position patching | `stage17_span.py capture`, `stage17_span.py`, then `stage17_table.py` (`data/span_table.txt`, `data/span_generations.txt`) |
| rolling patch held through generation | `stage18_roll.py capture`, `stage18_roll.py`, then `stage18_table.py` (`data/roll_table.txt`, `data/roll_generations.txt`) |
| refusal-vs-answer margin under increasing refusal prefill | `stage19_margin.py` then `stage19_table.py` (`data/margin_table.txt`, `data/margin_per_fact.txt`) |
| multilingual / multi-turn / lens at generated positions | `stage20_multilingual.py`, `stage21_multiturn.py`, `stage22_lens_gen.py` (`data/lens_gen_table.txt`) |
| non-fact inputs (arithmetic, translation, continuation, instruction, gibberish, empty) | `stage24_nonfact.py` then `stage24_table.py` (`data/nonfact_table.txt`, `data/nonfact_generations.txt`) |
| per-layer profiles of all three models on shared axes | `stage23_profiles.py` (`data/layer_profiles.txt`, `data/layer_profiles.png`) |
| training-variant selectivity sweep (LR, early stop, 3:1 retain, KL) | `stage25_selectivity.py` then `stage25_table.py` (`data/selectivity_table.txt`) |
| sweep extensions: B at steps 1-6, B at seeds 1-2 | `stage26_selectivity_ext.py` then `stage26_table.py` (`data/selectivity_ext_table.txt`) |

## Plots

`data/loss_curves.png` (both conditions), `data/probe_sweep.png` (33 layers × 3 positions, both eval
sets), `data/measurements_bar.png` (five measurements with 95% CIs), `data/layer_sweep.png`
(frozen probes vs layer, three models), `data/prefill.png`, `data/gate_direction_norms.png`,
`data/positive_control.png`, `data/interventions.png`, `data/relearning.png`, `data/layer_profiles.png`,
`data/score_distribution.png` (Stage 1
filter input) — regenerated by `stage6_plots.py` / `stage4_layer_sweep.py` / `stage1_plot.py`.

## Not in git

`activations/*.pt` and `runs/*/step-*/` checkpoints are excluded for size; they rebuild from the
scripts above. `data/counterfact.json` and `probes/base_sweep.joblib` ARE committed (see
Environment), so probe-based numbers reproduce without the ROME server or a re-cache.
