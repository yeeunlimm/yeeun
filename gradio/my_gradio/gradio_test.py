import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

model_name = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

def class_img(inp):
    inputs = processor(images=inp, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return label

iface = gr.Interface(
    fn = class_img,
    inputs = gr.Image(type='pil', label='이미지를 업로드하세요'),
    outputs = "text")

iface.launch()