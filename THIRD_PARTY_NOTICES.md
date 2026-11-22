# Third-Party Notices and Model Provenance

This document records the provenance, upstream source repositories, artifact hashes, and verified licensing terms for all third-party models, weights, tokenizers, and datasets utilized within **ObjectMemory Studio**.

---

## 1. Summary Matrix

| Component | Local File Path | Size | SHA-256 Checksum | Upstream Source | Verified License | Commercial Redistribution |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **YOLOv8n ONNX** | `models/yolov8n.onnx` | 12.8 MB | `e0bd17e5cb5aeccbb0cf89c9bf8b8425c5bf36f835ccd323dc8beeb235c9ce23` | Ultralytics (`ultralytics/yolov8`) | **AGPL-3.0** *(Ultralytics)* | Restricted (Requires Enterprise License) |
| **YOLO-World v2 Small** | `models/yolov8s-worldv2.onnx` | 51.1 MB | `381ced485b23ed8f06de3e82bb2745e1420c181c64f0a176784c34a959d550a1` | Tencent AI Lab / Ultralytics | **AGPL-3.0** / **GPL-3.0** | Restricted (Requires Enterprise License) |
| **CLIP Text Encoder (Quantized)** | `models/clip_text_quantized.onnx` | 64.5 MB | `73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a` | OpenAI CLIP (`openai/CLIP`) / HuggingFace Optimum | **MIT** | Permissive |
| **CLIP Tokenizer** | `models/clip_tokenizer.json` | 2.2 MB | `f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd` | OpenAI / HuggingFace (`openai/clip-vit-base-patch32`) | **MIT** | Permissive |
| **Real Workbench Dataset** | `data/real_workbench/` | — | Manifest: `fd93a3153ae16f4317d7ed157f64082b3e1969ac78606ce98af7611f13c540eb` | ObjectMemory Pilot Collection | **CC BY 4.0** | Permissive with Attribution |
| **Synthetic Pilot Frames** | `data/synthetic_workbench/` | — | Synthetic Generator Scripts | ObjectMemory Repository | **Apache-2.0** | Permissive |

---

## 2. Component Details & Licensing Evidence

### 2.1 YOLOv8n (`models/yolov8n.onnx`)
- **Upstream Project**: Ultralytics YOLOv8
- **Repository**: https://github.com/ultralytics/ultralytics
- **Author**: Ultralytics Inc.
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0).
- **Notice**: Ultralytics software and pretrained models are licensed under the AGPL-3.0. Under AGPL-3.0, any distributed application or network service utilizing this model must provide corresponding source code under AGPL-3.0. For commercial proprietary deployment without source disclosure, an Ultralytics commercial license is required.

### 2.2 YOLO-World v2 Small (`models/yolov8s-worldv2.onnx`)
- **Upstream Project**: YOLO-World: Real-Time Open-Vocabulary Object Detection
- **Original Authors**: Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan (Tencent AI Lab & Huazhong University of Science and Technology)
- **Upstream Repositories**:
  - https://github.com/AILab-CVC/YOLO-World (Original implementation: GPL-3.0)
  - https://github.com/ultralytics/ultralytics (YOLO-World v2 ONNX export integration: AGPL-3.0)
- **License**: **AGPL-3.0 / GPL-3.0**.
- **Notice**: Pretrained weights derived from COCO, Objects365, and GoldG pretraining. Strong copyleft terms apply; redistribution in proprietary commercial software is restricted.

### 2.3 Quantized CLIP Text Transformer (`models/clip_text_quantized.onnx`)
- **Upstream Project**: CLIP (Contrastive Language-Image Pre-Training)
- **Original Author**: OpenAI
- **Repository**: https://github.com/openai/CLIP
- **Export & Quantization**: ONNX export of `openai/clip-vit-base-patch32` text branch using HuggingFace Optimum, quantized to 8-bit integers (`int8`) via ONNX Runtime quantization toolchain.
- **License**: **MIT License**.
- **Permissibility**: Permissive for academic and commercial use with standard MIT copyright notice.

### 2.4 CLIP Tokenizer (`models/clip_tokenizer.json`)
- **Upstream Project**: HuggingFace Tokenizers / OpenAI CLIP
- **Source**: `openai/clip-vit-base-patch32` vocabulary and merges.
- **License**: **MIT License**.
- **Permissibility**: Permissive.

### 2.5 Real Workbench Dataset (`data/real_workbench/`)
- **Origin**: Publicly captured desk photography for ObjectMemory benchmarking.
- **Annotations**: `data/real_workbench/manifest.json`.
- **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Permissibility**: Free to share and adapt under attribution.

---

## 3. Compliance and Legal Guidelines

1. **Academic and Research Use**:
   - All components are fully usable for research, academic evaluation, and open-source benchmarking.
2. **Commercial Deployment Caution**:
   - If ObjectMemory Studio is deployed as a commercial SaaS or distributed binary, the use of `models/yolov8n.onnx` and `models/yolov8s-worldv2.onnx` triggers AGPL-3.0 copyleft requirements.
   - To make the product commercially permissive, future milestones should provide an optional inference adapter for Apache-2.0 or MIT-licensed open-vocabulary models (e.g., MobileCLIP, OWLv2 under Apache-2.0).
