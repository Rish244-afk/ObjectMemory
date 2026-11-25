# ObjectMemory Model Weights & Reproducible Export Guide

This directory stores ONNX inference weights used by `DetectorRegistry` (`src/objectmemory/detection/registry.py`).

## Model Inventory

| Registry Key | File Path | Size | Git Status | SHA-256 Checksum | Role |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `yolov8m` | `models/yolov8m.onnx` | 99.0 MB | Excluded (see export below) | `7B22EB3FB9D99C6CEE2F8E5B4471A774986A3BE883E2B4C2F560EE94C2CBEA09` | **Production default** detector |
| `yolov8n` | `models/yolov8n.onnx` | 12.2 MB | Tracked in Git | `e0bd17e5cb5aeccbb0cf89c9bf8b8425c5bf36f835ccd323dc8beeb235c9ce23` | Lightweight CPU alternative |
| `yolo26m` | `models/yolo26m.onnx` | 78.1 MB | Excluded (see export below) | `992d8f3a0ff47b0ce19a6b889f6ac2dc14c2088a955e7505775acead012fbbd8` | Optional ONNX Runtime alternative |
| `yolo-world` | `models/yolov8s-worldv2.onnx` | 48.8 MB | Excluded | `381ced485b23ed8f06de3e82bb2745e1420c181c64f0a176784c34a959d550a1` | Open-vocabulary alternative |
| *(CLIP text)* | `models/clip_text_quantized.onnx` | 61.5 MB | Excluded | `73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a` | Quantized CLIP text encoder |
| *(Tokenizer)* | `models/clip_tokenizer.json` | 2.1 MB | Tracked in Git | `f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd` | CLIP BPE tokenizer |

> **Why is `yolov8m.onnx` not committed directly?**  
> At 103,809,643 bytes (~99.0 MiB), `yolov8m.onnx` sits right against GitHub's 100 MB file-size limit. To keep repository clones fast and reliable without exceeding GitHub binary limits, `models/yolov8n.onnx` is tracked directly while `models/yolov8m.onnx` is reproduced from official Ultralytics weights using the command below.

## Reproducible Official Export for `yolov8m.onnx` and `yolo26m.onnx`

Requires `ultralytics` and `onnx` (`pip install ultralytics onnx`):

```powershell
# Default YOLOv8m
python -c "from ultralytics import YOLO; m = YOLO('yolov8m.pt'); m.export(format='onnx', imgsz=640, simplify=True)"
Move-Item yolov8m.onnx models/yolov8m.onnx -Force

# Optional YOLO26m (PyTorch SHA-256: 401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7)
python -c "from ultralytics import YOLO; m = YOLO('yolo26m.pt'); m.export(format='onnx', imgsz=640, opset=12, simplify=False)"
Move-Item yolo26m.onnx models/yolo26m.onnx -Force
```

On Linux/macOS:
```bash
python -c "from ultralytics import YOLO; m = YOLO('yolov8m.pt'); m.export(format='onnx', imgsz=640, simplify=True)"
mv yolov8m.onnx models/yolov8m.onnx
```

If `models/yolov8m.onnx` is not present locally, you can still run ObjectMemory immediately by selecting `yolov8n` in the web UI or API (`detector_name="yolov8n"`).

## Licensing

Ultralytics YOLOv8 and YOLO-World weights are governed by **AGPL-3.0** (and GPL-3.0 for original YOLO-World). See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for full provenance and compliance details.
