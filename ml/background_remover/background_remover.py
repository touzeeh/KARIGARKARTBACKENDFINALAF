import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BackgroundRemover:
    def __init__(self):
        print("Initializing KarigarKart Background Remover...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Background remover device: {self.device}")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite",
            trust_remote_code=True,
        )

        self.model.to(self.device)

        if self.device == "cpu":
            self.model.float()

        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        print("Background removal model loaded successfully")

    def remove_background(self, image: Image.Image) -> Image.Image:
        original_size = image.size

        image = image.convert("RGB")

        model_dtype = next(self.model.parameters()).dtype

        input_image = (
            self.transform(image)
            .unsqueeze(0)
            .to(device=self.device, dtype=model_dtype)
        )

        with torch.no_grad():
            prediction = self.model(input_image)[-1].sigmoid().cpu()

        mask = prediction[0].squeeze()

        mask = transforms.ToPILImage()(mask)

        mask = mask.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

        image = image.resize(original_size)

        image.putalpha(mask)

        return image