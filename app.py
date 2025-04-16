import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Multilabel-Portrait-SigLIP2"  # Replace with actual HF model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    0: "Anime Portrait",
    1: "Cartoon Portrait",
    2: "Real Portrait",
    3: "Sketch Portrait"
}

def classify_portrait(image):
    """Predict the type of portrait style from an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_portrait,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Portrait Type Prediction Scores"),
    title="Multilabel-Portrait-SigLIP2",
    description="Upload a portrait-style image (anime, cartoon, real, or sketch) to predict its most likely visual category."
)

if __name__ == "__main__":
    iface.launch()
