# knows-vs-says

Pilot: full fine-tune of Qwen/Qwen3.5-4B to respond "I don't know." on 53 CounterFact facts, with a
matched control fine-tune, and a frozen linear probe on base-model activations applied to both
fine-tuned models. Complete technical record: `data/RESULTS.md`. Final table:
`data/results_table.txt`.

## Environment

A100 80GB. `pip install torch torchvision torchaudio` (torch ≥ 2.5 required; pilot used
2.13.0+cu130), then `pip install transformers datasets accelerate scikit-learn matplotlib joblib`
(pilot: transformers 5.16.1, scikit-learn 1.9.0). Model and dataset download on first use. All
scripts run from the repo root; seed 0 throughout.

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

## Plots

`data/loss_curves.png` (both conditions), `data/probe_sweep.png` (33 layers × 3 positions, both eval
sets), `data/measurements_bar.png` (five measurements with 95% CIs), `data/layer_sweep.png`
(frozen probes vs layer, three models), `data/prefill.png`, `data/gate_direction_norms.png`,
`data/positive_control.png`, `data/interventions.png`, `data/score_distribution.png` (Stage 1
filter input) — regenerated by `stage6_plots.py` / `stage4_layer_sweep.py` / `stage1_plot.py`.

## Not in git

`data/counterfact.json` (stage0 re-downloads it), `activations/*.pt`, `probes/base_sweep.joblib`,
and `runs/*/step-*/` checkpoints are reproducible from the scripts above and excluded for size.
