# Summary

## Context

- Repo: `DEIM`
- Current custom task: fine-tuning DEIM/DFINE HGNetv2-X on a 9-class subset of `train_dataset`
- Dataset source format: YOLO-like
- Converter: `tools/dataset/yolo_to_coco.py`
- Active dataset config: `configs/dataset/train_dataset_detection_include_1_9.yml`
- Main training family:
  - `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_tuning.yml`
  - `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_tuning_768.yml`
  - `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_tuning_768_freeze_stage1.yml`
  - `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_noaug_768_unfreeze_backbone.yml`
  - `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_noaug_768_unfreeze_backbone_prodigyplus.yml`

## Dataset / Config Notes

- `tools/dataset/yolo_to_coco.py` supports two mutually exclusive filtering modes:
  - `--include-classes ...` keeps only listed classes
  - `--exclude-classes ...` keeps everything except listed classes
- The 9-class setup uses `configs/dataset/train_dataset_detection_include_1_9.yml` with `num_classes: 9`.
- The base 768 tuning config keeps `MixUp` off and uses `mosaic_prob: 0.1`.
- The `noaug` baseline is not literally zero augmentation: it keeps `RandomHorizontalFlip` and `Resize`, but removes mosaic, photometric distortions, zoom-out, and mixup.

## Important Code Changes Already Landed

- `tools/dataset/yolo_to_coco.py`
  - Added YOLO-like to COCO conversion for this repo.
  - Writes `instances_train.json`, `instances_val.json`, and `category_mapping.json`.
- `engine/solver/_solver.py`
  - Improved class-head remapping when loading checkpoints with mismatched class counts.
  - Handles `decoder.enc_score_head.*`, `decoder.dec_score_head.*`, and `decoder.denoising_class_embed.weight`.
  - Supports Obj365 -> COCO, COCO -> Obj365, and generic repeated-init fallback.
- `engine/deim/deim_criterion.py`
  - Fixed class-agnostic encoder matching/loss path for `query_select_method=agnostic`.
- `engine/data/transforms/_transforms.py`
  - `ConvertBoxes` and `ConvertPILImage` now correctly handle tuple-style samples in this pipeline.
- `engine/solver/det_solver.py`
  - `last.pth` is no longer rewritten every epoch; it is saved only periodically/finally.
  - `best_stg1.pth` is selected by `test_coco_eval_bbox[0]`.
- Metric meaning:
  - `test_coco_eval_bbox[0]` = COCO AP@[0.50:0.95]
  - `test_coco_eval_bbox[1]` = AP50
  - There is no built-in early stopping / patience here.

## Prodigy+ScheduleFree Integration

- Code touched:
  - `engine/optim/optim.py`
  - `engine/core/yaml_config.py`
  - `engine/core/workspace.py`
  - `engine/solver/det_engine.py`
  - `engine/solver/det_solver.py`
- What changed:
  - Registered `ProdigyPlusScheduleFree`.
  - Registered `ConstantLR`.
  - Allowed `lr_scheduler: ~` and `lr_warmup_scheduler: ~` without crashing.
  - Filtered unexpected kwargs in `create()` so merged YAML fields do not break constructors.
  - Added optimizer `train()` / `eval()` switching for schedule-free behavior during train vs validation.
  - Logged `Optim/d_pg_*` to TensorBoard.
- Current Prodigy config:
  - `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_noaug_768_unfreeze_backbone_prodigyplus.yml`
  - `lr=1.0`, `d0=1e-7`, constant LR, no warmup, no grad clipping
- TensorBoard group meaning for this config:
  - `Optim/d_pg_0` = encoder/decoder norm-BN group
  - `Optim/d_pg_1` = everything else, including backbone and the rest of the heads
- Runtime dependency:
  - Needs `prodigy-plus-schedule-free` installed in the venv.

## Latest Two Full Runs

Both latest full runs were 48-epoch `noaug + 768 + unfreeze backbone` baselines with validation every epoch.

| Run | Output dir | Best epoch | Best AP | Best AP50 | Best AP75 | Best AR100 | Last AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW baseline | `deim_outputs/baselines/full_noaug_unfreeze_seed0/output` | 19 | 0.4661 | 0.7167 | 0.5026 | 0.6628 | 0.4419 |
| Prodigy+SF baseline | `deim_outputs/baselines/full_noaug_unfreeze_prodigyplus_seed0/output` | 24 | 0.4913 | 0.7481 | 0.5120 | 0.6774 | 0.4891 |

Checkpoint sizes:

- AdamW `best_stg1.pth`: about `718M`
- Prodigy+SF `best_stg1.pth`: about `503M`

## Conclusions From The Two Latest Runs

- The task is learnable with this repo; we are not in the earlier "stuck near zero forever" regime anymore.
- The simple `768 + noaug + full backbone unfreeze` recipe is currently a valid baseline.
- Prodigy+ScheduleFree beat the AdamW baseline on the same recipe:
  - `+0.0251` AP
  - `+0.0314` AP50
  - `+0.0146` AR100
- Late-epoch stability was better with Prodigy+SF:
  - final AP at epoch 47 was `0.4891` vs `0.4419` for AdamW
  - delta at the end: `+0.0471` AP
- For the Prodigy run, `d` grew from `1e-7` to roughly:
  - `train_optim_d = 3.56e-3`
  - `train_optim_d_max = 7.06e-3`
  - no obvious catastrophic instability was observed

## Practical Restart Notes

- If resuming work, the best current baseline to build on is:
  - config: `configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_noaug_768_unfreeze_backbone_prodigyplus.yml`
  - run dir: `deim_outputs/baselines/full_noaug_unfreeze_prodigyplus_seed0/output`
  - best checkpoint: `deim_outputs/baselines/full_noaug_unfreeze_prodigyplus_seed0/output/best_stg1.pth`
- Validation is done in eval mode:
  - `model.eval()`
  - `criterion.eval()`
  - `optimizer.eval()` if the optimizer exposes it
- Training switches back to:
  - `model.train()`
  - `criterion.train()`
  - `optimizer.train()` if supported
- One caveat: `log.txt` does not record the original `-t` checkpoint path, so future launches should keep that noted explicitly in the run metadata or command log.
