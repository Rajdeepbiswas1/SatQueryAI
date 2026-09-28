import sys
import traceback
from pathlib import Path

import torch

PROJECT = Path(r"D:\SatQueryAI")
RECOVERED = PROJECT / "src" / "models" / "grounding_recovered"

POINT_SRC = RECOVERED / "point_grounding_recovered.py"
C3_SRC = RECOVERED / "c3_reference_grounding_recovered.py"

POINT_CKPT = PROJECT / "models" / "checkpoints" / "point_grounding" / "best.pt"
C3_CKPT = PROJECT / "models" / "checkpoints" / "c3_v1" / "best.pt"

print("=" * 72)
print("SATQUERYAI RECOVERED GROUNDING — LOCAL VERIFICATION")
print("=" * 72)
print("Device: CPU")
print()

# ------------------------------------------------------------
# 1. Verify files
# ------------------------------------------------------------
required = [
    POINT_SRC,
    C3_SRC,
    POINT_CKPT,
    C3_CKPT,
]

for path in required:
    print(f"{path}: {'FOUND' if path.exists() else 'MISSING'}")

missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Required files are missing:\n" +
        "\n".join(str(p) for p in missing)
    )

print()

# ------------------------------------------------------------
# 2. Import recovered modules directly from file paths
# ------------------------------------------------------------
def load_module(module_name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("Loading recovered Point source...")
try:
    point_mod = load_module("point_grounding_recovered", POINT_SRC)
    print("POINT SOURCE IMPORT: PASS")
except Exception:
    print("POINT SOURCE IMPORT: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

print("Loading recovered C3 source...")
try:
    c3_mod = load_module("c3_reference_grounding_recovered", C3_SRC)
    print("C3 SOURCE IMPORT: PASS")
except Exception:
    print("C3 SOURCE IMPORT: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

# ------------------------------------------------------------
# 3. Locate exact classes
# ------------------------------------------------------------
point_cls = getattr(point_mod, "PointGroundingModel", None)
c3_cls = getattr(c3_mod, "C3QueryConditionedGrounding", None)

print(
    "PointGroundingModel:",
    "FOUND" if point_cls is not None else "MISSING"
)

print(
    "C3QueryConditionedGrounding:",
    "FOUND" if c3_cls is not None else "MISSING"
)

if point_cls is None or c3_cls is None:
    raise RuntimeError("Recovered model class missing.")

print()

# ------------------------------------------------------------
# 4. Instantiate models
# ------------------------------------------------------------
print("Instantiating PointGroundingModel...")
try:
    point_model = point_cls()
    point_model.eval()
    point_params = sum(p.numel() for p in point_model.parameters())
    print(f"POINT INSTANTIATION: PASS")
    print(f"POINT PARAMS: {point_params:,}")
except Exception:
    print("POINT INSTANTIATION: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

print("Instantiating C3QueryConditionedGrounding...")
try:
    c3_model = c3_cls()
    c3_model.eval()
    c3_params = sum(p.numel() for p in c3_model.parameters())
    print(f"C3 INSTANTIATION: PASS")
    print(f"C3 PARAMS: {c3_params:,}")
except Exception:
    print("C3 INSTANTIATION: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

# ------------------------------------------------------------
# 5. Load checkpoints
# ------------------------------------------------------------
print("Loading Point checkpoint...")
try:
    point_ckpt = torch.load(
        POINT_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    point_state = point_ckpt["model_state_dict"]
    point_model.load_state_dict(point_state, strict=True)

    print("POINT CHECKPOINT LOAD: PASS")
    print(f"POINT CHECKPOINT EPOCH: {point_ckpt.get('epoch')}")
    print(
        f"POINT CHECKPOINT VAL mIoU: "
        f"{point_ckpt.get('val_mean_iou')}"
    )

except Exception:
    print("POINT CHECKPOINT LOAD: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

print("Loading C3 checkpoint...")
try:
    c3_ckpt = torch.load(
        C3_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    c3_state = c3_ckpt["model_state_dict"]
    c3_model.load_state_dict(c3_state, strict=True)

    print("C3 CHECKPOINT LOAD: PASS")
    print(f"C3 CHECKPOINT EPOCH: {c3_ckpt.get('epoch')}")
    print(f"C3 CHECKPOINT VAL IoU: {c3_ckpt.get('val_iou')}")

except Exception:
    print("C3 CHECKPOINT LOAD: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

# ------------------------------------------------------------
# 6. Parameter sanity against verified training versions
# ------------------------------------------------------------
EXPECTED_POINT_PARAMS = 1_531_909
EXPECTED_C3_PARAMS = 2_186_753

print("Parameter compatibility:")

if point_params == EXPECTED_POINT_PARAMS:
    print("POINT PARAMETER COUNT: PASS")
else:
    print(
        "POINT PARAMETER COUNT: FAIL "
        f"(expected {EXPECTED_POINT_PARAMS:,}, got {point_params:,})"
    )
    sys.exit(1)

if c3_params == EXPECTED_C3_PARAMS:
    print("C3 PARAMETER COUNT: PASS")
else:
    print(
        "C3 PARAMETER COUNT: FAIL "
        f"(expected {EXPECTED_C3_PARAMS:,}, got {c3_params:,})"
    )
    sys.exit(1)

print()

# ------------------------------------------------------------
# 7. Helper presence
# ------------------------------------------------------------
print("Checking C3 helper functions...")

for name in ("heatmap_moments", "heatmap_to_box"):
    present = callable(getattr(c3_mod, name, None))
    print(f"{name}: {'FOUND' if present else 'MISSING'}")
    if not present:
        raise RuntimeError(f"Missing required helper: {name}")

print()

# ------------------------------------------------------------
# 8. Smoke forward with synthetic features
# ------------------------------------------------------------
print("Running CPU smoke forwards...")

torch.manual_seed(42)

S1 = torch.randn(1, 196, 768)
S2 = torch.randn(1, 196, 768)
TEXT = torch.randn(1, 768)
POINT = torch.tensor([[0.50, 0.50]], dtype=torch.float32)

# Point
try:
    with torch.no_grad():
        point_out = point_model(S1, S2, TEXT, POINT)

    print("POINT FORWARD: PASS")
    print("Point output keys:", list(point_out.keys()))

    for key, value in point_out.items():
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Non-finite Point output detected: {key}"
                )

except Exception:
    print("POINT FORWARD: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

# C3
try:
    with torch.no_grad():
        c3_out = c3_model(S1, S2, TEXT)

    print("C3 FORWARD: PASS")
    print("C3 output keys:", list(c3_out.keys()))

    for key, value in c3_out.items():
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Non-finite C3 output detected: {key}"
                )

except Exception:
    print("C3 FORWARD: FAIL")
    traceback.print_exc()
    sys.exit(1)

print()

# ------------------------------------------------------------
# 9. Print important output shapes
# ------------------------------------------------------------
print("POINT OUTPUT SHAPES")
for key, value in point_out.items():
    if torch.is_tensor(value):
        print(f"  {key}: {tuple(value.shape)}")

print()

print("C3 OUTPUT SHAPES")
for key, value in c3_out.items():
    if torch.is_tensor(value):
        print(f"  {key}: {tuple(value.shape)}")

print()
print("=" * 72)
print("RECOVERED GROUNDING LOCAL VERIFICATION COMPLETE")
print("=" * 72)
print("TRAINING: NOT STARTED")
print("KAGGLE: NOT USED")
print("DEVICE: CPU")
print("=" * 72)