import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download


class BackgroundRemover:
    def __init__(self):
        print("Initializing KarigarKart Background Remover with ONNX...")

        self.input_size = 448

        model_path = hf_hub_download(
            repo_id="senty-au/BiRefNet_lite-ONNX-dynamic",
            filename="onnx/model.onnx",
        )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print("ONNX Background Remover loaded successfully")
        print(f"Input size: {self.input_size}x{self.input_size}")

    def remove_background(self, image: Image.Image) -> Image.Image:
        original_size = image.size

        image = image.convert("RGB")

        resized = image.resize(
            (self.input_size, self.input_size),
            Image.Resampling.BILINEAR,
        )

        x = np.asarray(
            resized,
            dtype=np.float32,
        ) / 255.0

        x = (
            x
            - np.array(
                [0.485, 0.456, 0.406],
                dtype=np.float32,
            )
        ) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

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