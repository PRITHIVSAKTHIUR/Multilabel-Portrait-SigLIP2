![zdfghgdftg.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/N8f6dbYHMmE02vq5wseAE.png)

# **Multilabel-Portrait-SigLIP2**

> **Multilabel-Portrait-SigLIP2** is a vision-language model fine-tuned from [**google/siglip2-base-patch16-224**](https://huggingface.co/google/siglip2-base-patch16-224) using the `SiglipForImageClassification` architecture. It classifies portrait-style images into one of the following **visual portrait categories**:

```py
 Classification Report:
                  precision    recall  f1-score   support

  Anime Portrait     0.9989    0.9991    0.9990      4444
Cartoon Portrait     0.9964    0.9926    0.9945      4444
   Real Portrait     0.9964    0.9971    0.9967      4444
 Sketch Portrait     0.9971    1.0000    0.9985      4444

        accuracy                         0.9972     17776
       macro avg     0.9972    0.9972    0.9972     17776
    weighted avg     0.9972    0.9972    0.9972     17776
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/vOq9EHhJGLzRJSQJ5_liQ.png)

---

# **Model Objective**

The model is designed to **analyze portrait images** and categorize them into **one of four distinct portrait types**:

- **0:** Anime Portrait  
- **1:** Cartoon Portrait  
- **2:** Real Portrait  
- **3:** Sketch Portrait

---

# **Try it with Transformers 🤗**

Install dependencies:

```bash
pip install -q transformers torch pillow gradio
```

Run the model with the following script:

```python
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
```

---

# **Intended Use Cases**

-  **AI Art Curation** — Automatically organize large-scale datasets of artistic portraits.  
-  **Style-based Portrait Analysis** — Determine artistic style in user-uploaded or curated portrait datasets.  
-  **Content Filtering for Platforms** — Group and recommend based on visual aesthetics.  
-  **Dataset Pre-labeling** — Helps reduce manual effort in annotation tasks.  
-  **User Avatar Classification** — Profile categorization in social or gaming platforms.
