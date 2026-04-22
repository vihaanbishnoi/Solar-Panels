"""
Converts your trained PyTorch model (.pth) to ONNX format (.onnx)
so it can run inside a web browser using ONNX Runtime Web.
"""

import torch
import timm
import onnx
import json
import os

LOAD_PATH   = "../artifacts/vision/model/solar_model_v1.pth"    # change to v2.pth after finetuning
ONNX_PATH   = "../artifacts/vision/model/model.onnx"           # output file (used by index.html)
NUM_CLASSES = 6
IMAGE_SIZE  = 224

# Must match the exact folder names and order ImageFolder used during training
# ImageFolder sorts alphabetically — verify this matches your training output
CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]
# ─────────────────────────────────────────────


def convert():
    device = torch.device("cpu")   # always convert on CPU

    # ── Check input file exists ──
    if not os.path.exists(LOAD_PATH):
        raise FileNotFoundError(
            f"Model file not found: {LOAD_PATH}\n"
            f"Run train.py first to create it."
        )

    # ── Load model ──
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    model.eval()   # must be in eval mode for export
    print(f"Loaded model from: {LOAD_PATH}")

    # ── Create dummy input (same shape as real images) ──
    # batch_size=1 means the browser sends one image at a time
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # ── Export to ONNX ──
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,        # save weights inside the onnx file
        opset_version=11,          # ONNX version (11 is widely supported)
        do_constant_folding=True,  # optimise the graph
        input_names=["image"],     # name of the input tensor
        output_names=["scores"],   # name of the output tensor
        dynamic_axes={
            "image":  {0: "batch_size"},   # allow variable batch size
            "scores": {0: "batch_size"}
        }
    )
    print(f"Exported ONNX model to: {ONNX_PATH}")

    # ── Verify the ONNX file is valid ──
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified — no errors found.")

    # ── Quantize to FP16 for the Web ──
    print("\nQuantizing model to FP16 to reduce web download size...")
    try:
        from onnxconverter_common import float16
        onnx_fp16 = float16.convert_float_to_float16(onnx_model)
        fp16_path = ONNX_PATH.replace(".onnx", "_fp16.onnx")
        onnx.save(onnx_fp16, fp16_path)
        print(f"Exported FP16 ONNX model to: {fp16_path}")
        final_model_path = fp16_path
    except ImportError:
        print("\nWarning: Please install `onnxconverter-common` for FP16 quantization.")
        print("Skipping FP16 quantization. Using base model.")
        final_model_path = ONNX_PATH

    # ── Save class names to JSON (used by index.html) ──
    # This way your website always knows which class is which
    labels_path = "class_names.json"
    with open(labels_path, "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"Class names saved to: {labels_path}")

    # ── Print file info ──
    size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
    print(f"\nModel size: {size_mb:.1f} MB")
    print(f"\nFiles ready for your website:")
    print(f"  {final_model_path}")
    print(f"  {labels_path}")
    print(f"\nCopy both files into your website folder alongside index.html.")


if __name__ == "__main__":
    convert()