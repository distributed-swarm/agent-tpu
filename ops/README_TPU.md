# TPU Ops (Edge TPU / Coral)

This folder contains TPU-backed ops for the agent.

Rules:
- TPU ops stay small and pure
- INT8 TFLite only
- No preprocessing or batching here
- CPU fallback handled elsewhere

Initial ops:
- map_classify_tpu (stub)
