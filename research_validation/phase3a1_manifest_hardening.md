# ObjectMemory: Phase 3A.1 Manifest Loader Hardening Report
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Phase:** 3A.1 (Evaluation Manifest Loader Hardening)  
**Status:** IMPLEMENTED & TESTED (32 / 32 focused tests passing)

---

## 1. Overview & Objectives

Phase 3A.1 hardens `src/objectmemory/benchmarks/manifest.py` against malformed data, path traversal, ambiguous provenance, and silent omission bugs prior to building downstream evaluation components.

All changes were strictly confined to:
1. `src/objectmemory/benchmarks/manifest.py` (hardened loader and models).
2. `tests/test_benchmark_manifest.py` (extended test suite with 20 focused regression tests).
3. Documentation artifacts.

**Scope Confirmation:** Zero modifications were made to the temporal engine (`src/objectmemory/temporal/`), registration logic (`src/objectmemory/registration/`), database schemas (`src/objectmemory/persistence/`), detection models, or production application behavior.

---

## 2. Exact Changes Made

### A. Mandatory Image Path Validation
- Every observation must have a non-empty, non-null string `image_path`.
- Missing, null, or whitespace-only image paths now generate an explicit contextual error:
  `[{seq_id} / {o_id}] Missing or empty required 'image_path'`.
- Observations are never silently dropped; every declared observation is represented in the sequence queue, and any invalid path causes `ManifestValidationError` to be raised at load time.

### B. Strict Object Annotation Validation (`ObjectAnnotationEntry.from_dict`)
- **Safe Parsing:** Uses `from_dict(data, scope)` returning `Tuple[Optional[ObjectAnnotationEntry], List[str]]`. Never allows uncaught `KeyError`, `TypeError`, or `ValueError` to crash the loader.
- **Identifier & Label Validation:** Asserts non-empty `object_id` and non-empty `class_name` (or `class_label`).
- **Bounding Box Rigor:**
  - Validates exact coordinate count (exactly 4 coordinates).
  - Asserts numeric and strictly finite values (rejects booleans, strings, `NaN`, and `inf`).
  - Asserts geometric validity: `x_min < x_max` and `y_min < y_max` (rejects zero-area, inverted, or collapsed boxes).
  - Supports both 4-element sequences (`[x_min, y_min, x_max, y_max]`) and coordinate dictionaries (`{"x_min": ..., ...}`).
- **Vocabulary Enforcement:**
  - **Visibility:** Validated against canonical `VisibilityStatus` values (`CLEARLY_VISIBLE`, `PARTIALLY_OCCLUDED`, `HEAVILY_OCCLUDED`, `OUT_OF_FRAME`, `UNOBSERVED`, `UNCERTAIN`). Explicit legacy mappings supported: `"VISIBLE" -> "CLEARLY_VISIBLE"`, `"FULLY_OCCLUDED" -> "HEAVILY_OCCLUDED"`, `"ABSENT" -> "UNOBSERVED"`. Unrecognized strings generate validation errors.
  - **Epistemic States:** Validated against canonical `EpistemicState` values (`CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `CONFIRMED_REMOVED`, `CONFIRMED_ADDED`, `PENDING_REMOVAL`, `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_IDENTITY`). Explicit legacy mappings supported: `"MOVED" -> "CONFIRMED_MOVED"`, `"MOVED_BACK_TO_BASELINE" -> "CONFIRMED_PRESENT"`, `"STATIONARY" -> "CONFIRMED_PRESENT"`, `"REMOVED" -> "CONFIRMED_REMOVED"`, `"REAPPEARED" -> "CONFIRMED_PRESENT"`, `"ADDED" -> "CONFIRMED_ADDED"`. Unrecognized states generate validation errors.
- **Confidence Bounds:** Asserts `annotation_confidence` is a finite float in $[0.0, 1.0]$.
- **Metadata Preservation:** All unrecognized annotation keys are retained in `metadata`.

### C. Deterministic Dataset-Root Resolution
Resolution follows a strict, unambiguous precedence order:
1. **Explicit Argument:** An explicit `dataset_root` argument passed to `load()` or `parse_dict()` takes absolute precedence. If it does not exist or is not a directory, raises a clear error.
2. **Manifest Declared Root:** If manifest contains `dataset_root`, resolves relative to manifest file parent (if relative) or directly (if absolute). If it does not exist or is not a directory, raises a clear error instead of silently falling back.
3. **Manifest Parent Directory:** If neither is specified, resolves to the manifest file's parent directory.
4. **Current Working Directory:** Fallback for in-memory dictionary parsing without an associated file path.
5. **Symlink and Path Traversal Enforcement:** All resolved paths (and symlink targets) must remain contained within `dataset_root.resolve()`. Any path escaping the root triggers `Path traversal violation detected`.

### D. Extended File Validation (`DatasetManifest.validate_files`)
Validates physical existence across all supported reference types when enabled:
- Observation primary images (`check_images=True`).
- Sequence reference images (`check_images=True`).
- Sequence calibration files (`check_calibration=True`).
- Sequence ground-truth files (`check_annotations=True`).
- Observation annotation files (`check_annotations=True`).
- Observation metric depth maps (`check_depth=True`).
- Observation semantic masks (`check_semantics=True`).
*Scope Notice:* Validates on-disk existence only. Content parsing of external annotation files is not asserted.

### E. Explicit Provenance Classification
Replaced fragile substring matching with an explicit `DatasetProvenance` enum:
- `GENUINE_PHYSICAL`
- `SYNTHETIC_GENERATED`
- `EXPLORATORY_PILOT`
- `PUBLIC_BENCHMARK`
- `UNKNOWN`
Manifests with missing or unrecognized provenance default strictly to `DatasetProvenance.UNKNOWN` and are **never** represented as verified physical data.

### F. Timestamp Normalization & Ordering
- Supports timezone-aware ISO-8601 strings and Unix epoch numeric timestamps (documented accepted alternative).
- Normalizes all timestamps to UTC datetime objects for monotonic chronological comparison.
- **Equal Timestamps:** Observations with identical timestamps ($t_k = t_{k+1}$) are permitted (e.g. synchronized stereo or multi-camera rigs) and preserve their declared manifest order.
- **Non-Chronological Rejection:** Any observation with $t_{k+1} < t_k$ generates a validation error; observations are never silently reordered.

---

## 3. Test Suite Verification

### Focused Unit & Regression Test Suite
Executed command:
```powershell
Set-Location "C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory"; $env:PYTHONPATH="src"; python -m unittest tests.test_benchmark_manifest
```
**Result:**
```
................s...............
----------------------------------------------------------------------
Ran 32 tests in 0.314s

OK (skipped=1)
```
*(31 passed, 1 skipped: symlink privilege check gracefully skipped where OS user lacks administrative symlink creation privileges).*

### Complete Repository Test Suite
Executed command:
```powershell
Set-Location "C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory"; $env:PYTHONPATH="src"; python -m unittest discover -s tests -p "test_*.py"
```
**Result:**
```
----------------------------------------------------------------------
Ran 369 tests in 158.933s

OK (skipped=1)
```
*(368 passed, 1 skipped: 337 pre-existing baseline unit tests + 32 focused manifest loader tests with zero regressions).*

---

## 4. Unresolved Limitations & Upstream Guarantees

1. **External Annotation Parsing:** `validate_files()` asserts the physical presence of external JSON files referenced by `source_annotation_path`, but does not recursively parse or validate their internal schemas.
2. **Next Component Boundary:** The manifest loader provides an immutable, strictly validated `DatasetManifest` with explicit provenance. The upcoming `runner.py` in Phase 3C must check `manifest.is_genuine_physical` or `manifest.provenance` to enforce benchmark data exclusion rules.
