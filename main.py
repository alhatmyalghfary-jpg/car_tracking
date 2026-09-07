import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model

from xai import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap


@st.cache_resource
def load_classifier():
    inputs = tf.keras.Input(shape=(150, 150, 3))
    base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=(150, 150, 3),
    )
    features = base_model(inputs)
    x = Flatten()(features)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.load_weights("vehicle_classifier.h5")
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
    st.image(image, caption="الصورة التي تم رفعها")

    img_resized = image.resize((150, 150))
    img_array = np.expand_dims(np.asarray(img_resized, dtype=np.float32) / 255.0, axis=0)
    predictions = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0, pred_index])

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)
    st.image(
        overlay_heatmap(image, heatmap),
        caption=f"خريطة Grad-CAM - الطبقة: {last_conv_layer.name}",
    )

    if confidence < 0.9:
        st.write("الفئة: غير معروفة")
    else:
        st.write(f"الفئة المتوقعة: {categories[pred_index]}")
        st.write(f"درجة الثقة: {confidence * 100:.2f}%")