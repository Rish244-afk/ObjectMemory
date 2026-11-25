# ObjectMemory Datasets & Setup Guide

## Included Datasets (Tracked in Git)
- **`data/physical_pilot_v2/`**: 6-scenario, 30-observation physical tabletop pilot (`stationary`, `movement`, `removal`, `addition`, `occlusion`, `return_to_baseline`) with ground-truth annotations in `manifest_v2.json`.
- **`data/synthetic_workbench_v2/`**: 25-observation synthetic workbench benchmark (`S01`–`S05`) with `manifest.json`.
- **`data/real_workbench/`**: 7-image genuine physical desk pilot and manifest.
- **`data/controlled_temporal_v1/`**, **`data/physical_workbench_v1/`**, **`data/workbench_samples/`**: Supporting benchmark/sample fixtures.

## Excluded Large / Local Directories (Not Committed)
- **`data/eth_asl_lounge/`** (~19.6 GB): Third-party ETH ASL Lounge RGB-D & TSDF mesh sessions (`observation_0`–`observation_9`) used for supplementary 2D planar registration experiments. Metadata and evaluation results are preserved in `research_validation/eth_asl_lounge/`.
- **`data/workspaces/`**: Local SQLite memory databases (`memory.db`) and uploaded session images generated at runtime.
