# map_classify_tpu — Contract (v0)

## Purpose
Run INT8 TFLite classification on an Edge TPU.

## Inputs (payload)
- `input`: list[int]
  Flattened INT8 tensor matching model input shape.
- `model_path` (optional): string
  Absolute path to `*_edgetpu.tflite`.
- `topk` (optional): int
  Default: 5

## Outputs
- `op`: string (`map_classify_tpu`)
- `model_path`: string
- `topk`: list of objects:
  - `index`: int
  - `score`: float
- `elapsed_ms`: float

## Notes
- No preprocessing performed here
- No batching
- No label mapping
- CPU fallback handled elsewhere
