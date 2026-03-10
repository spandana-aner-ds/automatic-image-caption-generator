import streamlit as st
import numpy as np
from PIL import Image

# ---------------- Title ----------------
st.title("Image Caption Generator")
st.write("Upload an image to generate image-based captions .")

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def analyze_image(img):
    img_np = np.array(img)

    # Average color
    avg_color = img_np.mean(axis=(0, 1))
    r, g, b = avg_color

    if r > g and r > b:
        color_desc = "reddish tones"
    elif g > r and g > b:
        color_desc = "greenish tones"
    else:
        color_desc = "bluish tones"

    # Brightness
    brightness = avg_color.mean()
    if brightness > 160:
        light_desc = "bright"
    elif brightness > 90:
        light_desc = "moderately lit"
    else:
        light_desc = "dark"

    # Orientation
    width, height = img.size
    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "square"

    return color_desc, light_desc, orientation

def generate_captions(color, light, orientation):
    return [
        f"A {light} {orientation} image dominated by {color}.",
        f"This {orientation} photo features {color} and a {light} atmosphere.",
        f"A visually appealing {orientation} scene with noticeable {color}.",
        f"The image appears {light} and shows strong {color}.",
        f"A {orientation} photograph captured with {color} in the scene."
    ]

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    color, light, orientation = analyze_image(img)
    captions = generate_captions(color, light, orientation)

    st.subheader("Generated Captions")
    for i, cap in enumerate(captions, start=1):
        st.write(f"{i}. {cap}")
