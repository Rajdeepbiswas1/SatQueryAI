from __future__ import annotations

import numpy as np


def aggregate_area_conserving(
    src: np.ndarray,
    out_h: int = 56,
    out_w: int = 56,
) -> np.ndarray:
    """
    Exact area-weighted aggregation from source pixels to output cells.

    Each output cell represents an equal-area region of the normalized
    [0, 1] x [0, 1] patch.

    The returned values are area fractions in [0, 1].
    """

    src = np.asarray(src, dtype=np.float64)

    if src.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {src.shape}")

    h, w = src.shape

    # Normalized pixel boundaries.
    src_y = np.linspace(0.0, 1.0, h + 1)
    src_x = np.linspace(0.0, 1.0, w + 1)
    dst_y = np.linspace(0.0, 1.0, out_h + 1)
    dst_x = np.linspace(0.0, 1.0, out_w + 1)

    wy = np.zeros((out_h, h), dtype=np.float64)
    wx = np.zeros((out_w, w), dtype=np.float64)

    for oy in range(out_h):
        a0 = dst_y[oy]
        a1 = dst_y[oy + 1]

        for sy in range(h):
            b0 = src_y[sy]
            b1 = src_y[sy + 1]

            overlap = max(0.0, min(a1, b1) - max(a0, b0))

            if overlap > 0:
                wy[oy, sy] = overlap / (a1 - a0)

    for ox in range(out_w):
        a0 = dst_x[ox]
        a1 = dst_x[ox + 1]

        for sx in range(w):
            b0 = src_x[sx]
            b1 = src_x[sx + 1]

            overlap = max(0.0, min(a1, b1) - max(a0, b0))

            if overlap > 0:
                wx[ox, sx] = overlap / (a1 - a0)

    result = wy @ src @ wx.T

    # Remove microscopic floating-point excursions.
    result = np.clip(result, 0.0, 1.0)

    return result


def check(name: str, src: np.ndarray) -> None:
    out = aggregate_area_conserving(src, 56, 56)

    src_area = float(src.sum())
    out_area_equivalent = float(out.sum()) * (
        src.shape[0] / 56.0
    ) * (
        src.shape[1] / 56.0
    )

    error = abs(src_area - out_area_equivalent)

    print(f"\n{name}")
    print("-" * 60)
    print("source shape:", src.shape)
    print("output shape:", out.shape)
    print("source area:", src_area)
    print("output-equivalent area:", out_area_equivalent)
    print("absolute area error:", error)
    print("output min:", float(out.min()))
    print("output max:", float(out.max()))

    assert out.shape == (56, 56)
    assert out.min() >= -1e-12
    assert out.max() <= 1.0 + 1e-12
    assert error < 1e-10

    print("PASS")


def main() -> None:
    h = 120
    w = 120

    # 1. All zero
    check(
        "TEST 1 — ALL ZERO",
        np.zeros((h, w), dtype=np.float64),
    )

    # 2. All one
    check(
        "TEST 2 — ALL ONE",
        np.ones((h, w), dtype=np.float64),
    )

    # 3. Single source pixel
    single = np.zeros((h, w), dtype=np.float64)
    single[60, 60] = 1.0
    check("TEST 3 — SINGLE PIXEL", single)

    # 4. Rectangular region
    rectangle = np.zeros((h, w), dtype=np.float64)
    rectangle[20:80, 30:100] = 1.0
    check("TEST 4 — RECTANGLE", rectangle)

    # 5. Checkerboard
    yy, xx = np.indices((h, w))
    checkerboard = ((yy + xx) % 2).astype(np.float64)
    check("TEST 5 — CHECKERBOARD", checkerboard)

    # 6. Random binary mask
    rng = np.random.default_rng(12345)
    random_mask = (
        rng.random((h, w)) > 0.63
    ).astype(np.float64)
    check("TEST 6 — RANDOM BINARY MASK", random_mask)

    print("\n" + "=" * 60)
    print("ALL C3 AGGREGATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()