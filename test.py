import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from xai import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap

model_path = "vehicle_classifier.h5"  
model = tf.keras.models.load_model(model_path, compile=False)
model.summary()
last_conv_layer = find_last_conv_layer(model)
print(f"آخر طبقة تلافيفية مستخدمة في Grad-CAM: {last_conv_layer.name}")

#model = tf.keras.models.load_model("model.h5", safe_mode=False)
#model = tf.keras.models.load_model("model.h5", compile=False)


# الفئات المعروفة
categories = ["Bikes", "Cars", "Motorcycles"]

st.title("تصنيف صورة مركبة ")

uploaded_file = st.file_uploader("ارفع صورة مركبة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة وتحويلها لمصفوفة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة التي تم رفعها", use_container_width=True)    
    # معالجة الصورة لتتناسب مع المدخلات المطلوبة (150x150)
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد batch

    # توقع الفئة
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer,
        class_index=pred_index,
    )
    gradcam_image = overlay_heatmap(image, heatmap)
    st.image(
        gradcam_image,
        caption=f"خريطة Grad-CAM - الطبقة: {last_conv_layer.name}",
        use_container_width=True,
    )

    # شرط عدم معرفة الفئة إذا الثقة منخفضة (مثلاً أقل من 0.5)
    if confidence < 0.9:
        st.write("❓ الفئة: غير معروفة")
    else:
        st.write(f"✅ الفئة المتوقعة: {categories[pred_index]}")
        st.write(f"درجة الثقة: {confidence*100:.2f}%")
