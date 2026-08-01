import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained MobileNetV2 (fast, accurate enough for general use)
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()  # inference mode, not training

# The category labels that came with the weights (1000 ImageNet classes)
categories = weights.meta["categories"]

# The exact preprocessing the model was trained with
preprocess = weights.transforms()

def classify(image):
    if image is None:
        return {}
    img = image.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    top5 = torch.topk(probs, 5)
    return {categories[idx]: float(probs[idx]) for idx in top5.indices}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Recognition",
    description="Upload an image and the model will tell you what it is.",
)

if __name__ == "__main__":
    demo.launch()