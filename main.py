import time
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from config import HF_API_KEY

MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
]

HEADERS = {
    "Authorization" : f"Bearer {HF_API_KEY}",
    "Accept" : "image/png"
}

def generate_image(prompt):
    payload = {"inputs" : prompt}
    error = None

    for model in MODELS:
        url = f"https://router.huggingface.co/hf-inference/models/{model}"

        for _ in range(3):
            response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
            content_type = response.headers.get("content-type", "").lower()

            if response.status_code == 503:
                wait = response.json().get("estimated time", 5)
                print(f"Model loading ... waiting {wait}s.")
                time.sleep(wait + 1 )
                continue

            if response.ok and "image" in content_type:
                return Image.open(BytesIO(response.content)).convert("RGB")

            try:
                error = response.json()
            except:
                error = response.text
            break
    raise Exception(error)

def enhance_image(image):
    image = ImageEnhance.Brightness(image).enhance(1.2) #increase 20% brightness
    image = ImageEnhance.Contrast(image).enhance(1.3)
    return image.filter(ImageFilter.GaussianBlur(2))

def main():
    print("\n =========AI Image Generetaor=========")

    while True:
        prompt = input("\nDescribe an image('exit' to quit): ").strip()

        if prompt.lower() == "exit":
            break
        try:
            image = enhance_image(generate_image(prompt))
            image.show()

            if input("save image?(y/n):").lower() == "y":
                filename = input("Filename:").strip() + ".png"
                image.save(filename)
                print("saved as", filename)
        except Exception as e:
            print("Error:", e)
main()
