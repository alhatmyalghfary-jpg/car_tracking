import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model

from xai import (
    find_last_conv_layer,
    heatmap_to_image,
    make_gradcam_heatmap,
    overlay_heatmap,
)


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
    predicted_category = categories[pred_index]

    if confidence < 0.9:
        st.warning("الثقة أقل من 90%؛ لذلك اعتُبرت الفئة غير معروفة.")
    else:
        st.success(f"الفئة المتوقعة: {predicted_category}")
        st.write(f"درجة الثقة: {confidence * 100:.2f}%")

    st.markdown("### XAI")
    show_explanation = st.button("XAI")

    if show_explanation:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)
        heatmap_image = heatmap_to_image(heatmap, image.size)
        overlay_image = overlay_heatmap(image, heatmap)

        st.subheader("تفسير قرار النموذج بالصورة الحرارية")
        st.write(
            f"اختار النموذج فئة **{predicted_category}** لأنها حصلت على أعلى احتمال "
            f"({confidence * 100:.2f}%). الصور التالية توضح المناطق التي ركّز عليها "
            "النموذج عند اتخاذ القرار."
        )

        st.markdown("#### صور التفسير الحراري")
        image_columns = st.columns(3)
        with image_columns[0]:
            st.image(image, caption="1. الصورة المختبرة")
        with image_columns[1]:
            st.image(
                heatmap_image,
                caption="2. الخريطة الحرارية",
            )
        with image_columns[2]:
            st.image(
                overlay_image,
                caption="3. الصورة مع التفسير",
            )

        st.markdown("#### كيف تم اختيار الصنف؟")
        ranked_predictions = sorted(
            zip(categories, predictions[0]),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        for rank, (category, probability) in enumerate(ranked_predictions, start=1):
            probability_value = float(probability)
            st.write(f"{rank}. **{category}**: {probability_value * 100:.2f}%")
            st.progress(probability_value)

        st.info(
            f"الخريطة ناتجة عن طبقة **{last_conv_layer.name}** باستخدام Grad-CAM. "
            "الأحمر والأصفر يمثلان مناطق ذات تأثير أكبر على اختيار الصنف، "
            "والأزرق يمثل تأثيرًا أقل. هذه المناطق توضّح تركيز النموذج، "
            "وليست دليلًا قطعيًا على أن النموذج تعرّف على جزء محدد من المركبة."
        )