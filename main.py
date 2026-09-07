import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from xai import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap


@st.cache_resource
def load_classifier():
    model = tf.keras.models.load_model("vehicle_model.keras", compile=False)
    return model, find_last_conv_layer(model)


model, last_conv_layer = load_classifier()
categories = ["Bikes", "Cars", "Motorcycles"]

st.title("تصنيف صورة مركبة")
uploaded_file = st.file_uploader(
    "ارفع صورة مركبة",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة التي تم رفعها", use_container_width=True)

    img_resized = image.resize((150, 150))
    img_array = np.expand_dims(np.asarray(img_resized, dtype=np.float32) / 255.0, axis=0)
    predictions = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0, pred_index])

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)
    st.image(
        overlay_heatmap(image, heatmap),
        caption=f"خريطة Grad-CAM - الطبقة: {last_conv_layer.name}",
        use_container_width=True,
    )

    if confidence < 0.9:
        st.write("الفئة: غير معروفة")
    else:
        st.write(f"الفئة المتوقعة: {categories[pred_index]}")
        st.write(f"درجة الثقة: {confidence * 100:.2f}%")