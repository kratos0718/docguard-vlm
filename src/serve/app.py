"""Minimal live demo: upload a document image, pick OCR or forgery-check, see the
fine-tuned model's output. This is the thing you actually run in an interview
instead of reading numbers off a README.

    python src/serve/app.py --adapter outputs/qwen2vl-2b-docguard-lora

Requires the GPU eval deps (torch/transformers/peft) and, for the UI, `gradio`.
Reuses eval/evaluate.py's model-loading and generation code so the demo and the
scored eval harness are guaranteed to behave identically.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen.prompts import FORGERY_INSTRUCTIONS, OCR_INSTRUCTIONS
from eval.evaluate import build_model, generate

TASK_PROMPTS = {
    "OCR / field extraction": OCR_INSTRUCTIONS[0],
    "Forgery detection": FORGERY_INSTRUCTIONS[0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="unsloth/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter; omit to demo the zero-shot base model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()

    import gradio as gr
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"loading model on {device} (adapter={args.adapter})...")
    model, processor = build_model(args.base_model, args.adapter, device)

    def run(image, task, custom_instruction):
        if image is None:
            return "Upload a document image first."
        instruction = custom_instruction.strip() or TASK_PROMPTS[task]
        return generate(model, processor, image.convert("RGB"), instruction, device)

    with gr.Blocks(title="DocGuard-VLM") as demo:
        gr.Markdown(
            "# DocGuard-VLM\n"
            f"Qwen2-VL-2B{' + LoRA adapter' if args.adapter else ' (zero-shot baseline)'} — "
            "document field extraction and forgery detection."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Document image")
                task_in = gr.Radio(list(TASK_PROMPTS.keys()), value="Forgery detection", label="Task")
                custom_in = gr.Textbox(label="Custom instruction (optional, overrides task preset)")
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="Model output", lines=8)
        run_btn.click(run, inputs=[image_in, task_in, custom_in], outputs=out)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
