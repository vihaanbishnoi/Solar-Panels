"""
convert.py
----------
Converts your trained PyTorch model (.pth) to ONNX format (.onnx)
so it can run inside a web browser using ONNX Runtime Web.

Run this after train.py or finetune.py whenever you want to update
the browser model.

Requires:
    pip install torch torchvision timm onnx

Usage:
    python convert.py
"""

import torch
import timm
import onnx
import json
import os

# ─────────────────────────────────────────────
# CONFIGURE THESE
# ─────────────────────────────────────────────
LOAD_PATH   = "solar_model_v1.pth"    # change to v2.pth after finetuning
ONNX_PATH   = "model.onnx"           # output file (used by index.html)
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

    # ── Save class names to JSON (used by index.html) ──
    # This way your website always knows which class is which
    labels_path = "class_names.json"
    with open(labels_path, "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"Class names saved to: {labels_path}")

    # ── Print file info ──
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"\nModel size: {size_mb:.1f} MB")
    print(f"\nFiles ready for your website:")
    print(f"  {ONNX_PATH}")
    print(f"  {labels_path}")
    print(f"\nCopy both files into your website folder alongside index.html.")

    # ── Quick sanity check: run one prediction ──
    print("\nRunning sanity check...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(ONNX_PATH)
    dummy_np = dummy_input.numpy()
    outputs  = session.run(["scores"], {"image": dummy_np})
    scores   = outputs[0][0]

    # Convert raw scores to probabilities using softmax
    exp_scores = np.exp(scores - scores.max())
    probs      = exp_scores / exp_scores.sum()

    print("Output probabilities for dummy input:")
    for name, prob in zip(CLASS_NAMES, probs):
        print(f"  {name:25s}  {prob*100:.2f}%")

    print("\nSanity check passed. Model is working correctly.")
    print("\nDone! Next step: open index.html in your browser.")


if __name__ == "__main__":
    convert()