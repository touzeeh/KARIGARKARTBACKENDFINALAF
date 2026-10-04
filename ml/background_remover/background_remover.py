import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download


class BackgroundRemover:
    def __init__(self):
        print("Initializing KarigarKart Background Remover with FP16 ONNX...")

        model_path = hf_hub_download(
            repo_id="studioludens/birefnet-lite-512",
            filename="onnx/model_fp16.onnx",
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 512

        print("FP16 ONNX Background Remover loaded successfully")

    def remove_background(self, image: Image.Image) -> Image.Image:
        original_size = image.size

        image = image.convert("RGB")

        resized = image.resize(
            (512, 512),
            Image.Resampling.BILINEAR,
        )

        x = np.asarray(resized, dtype=np.float32) / 255.0

        x = (
            x - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        x = x.transpose(2, 0, 1)[None]

        logits = self.session.run(
            None,
            {self.input_name: x},
        )[0][0, 0]

        alpha = 1.0 / (1.0 + np.exp(-logits))

        mask = Image.fromarray(
            (alpha * 255).clip(0, 255).astype(np.uint8)
        )

        mask = mask.resize(
            original_size,
            Image.Resampling.BILINEAR,
        )

        image.putalpha(mask)

        return image