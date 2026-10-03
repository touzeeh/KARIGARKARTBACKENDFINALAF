from PIL import Image
from ml.background_remover.background_remover import BackgroundRemover


remover = BackgroundRemover()

image = Image.open("test.jpg")

result = remover.remove_background(image)

result.save("test_no_background.png")

print("Background removed successfully!")