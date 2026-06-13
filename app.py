import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

from model import CNN

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

CLASS_EMOJI = {
    "airplane":    "✈️",
    "automobile":  "🚗",
    "bird":        "🐦",
    "cat":         "🐱",
    "deer":        "🦌",
    "dog":         "🐶",
    "frog":        "🐸",
    "horse":       "🐴",
    "ship":        "🚢",
    "truck":       "🚛",
}

CHECKPOINT = "cnn_cifar10.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


@st.cache_resource
def load_model():
    model = CNN(num_classes=10)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    else:
        st.warning("No trained model found. Run `python train.py` first.")
    model.eval()
    return model.to(DEVICE)


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 CIFAR-10 Image Classifier")
st.markdown(
    "Custom CNN built **from scratch** in PyTorch — no pretrained weights. "
    "Upload any image and the model predicts one of 10 classes."
)

st.markdown("**Supported classes:** " +
            " ".join(f"`{c}`" for c in CLASSES))

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        model = load_model()
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx] * 100

        emoji = CLASS_EMOJI[pred_class]
        st.markdown(f"### {emoji} **{pred_class.capitalize()}**")
        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("**All class probabilities**")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["#4f8ef7" if i != pred_idx else "#22c55e"
                  for i in range(len(CLASSES))]
        ax.barh(CLASSES, probs * 100, color=colors)
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

st.divider()
st.markdown(
    "<small>Built by Devanshi Tiwari — "
    "[GitHub](https://github.com/Devanshi-cmd)</small>",
    unsafe_allow_html=True,
)