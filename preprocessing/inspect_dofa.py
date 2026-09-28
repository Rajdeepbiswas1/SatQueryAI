import inspect

from torchgeo.models import (
    dofa_base_patch16_224,
    DOFABase16_Weights,
)


def main():
    print("=" * 70)
    print("DOFA API INSPECTION")
    print("=" * 70)

    print("\nConstructor:")
    print(inspect.signature(dofa_base_patch16_224))

    model = dofa_base_patch16_224(
        weights=None
    )

    print("\nModel forward signature:")
    print(inspect.signature(model.forward))

    print("\nModel class:")
    print(type(model))

    print("\nModel:")
    print(model)


if __name__ == "__main__":
    main()