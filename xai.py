import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def find_last_conv_layer(model):
    """Return the last convolutional layer, including nested models."""
    convolutional_layers = [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not convolutional_layers:
        raise ValueError("لم يتم العثور على طبقة Conv2D في النموذج")
    return convolutional_layers[-1]


def make_gradcam_heatmap(image_batch, model, last_conv_layer, class_index=None):
    """Generate a normalized Grad-CAM heatmap for one image batch."""
    gradient_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        convolution_output, predictions = gradient_model(image_batch)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, convolution_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    convolution_output = convolution_output[0]
    pooled_gradients = pooled_gradients[0]
    weighted_output = convolution_output * pooled_gradients
    heatmap = tf.reduce_sum(weighted_output, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, opacity=0.4):
    """Overlay a colored heatmap on the original PIL image."""
    heatmap_array = np.uint8(255 * heatmap)
    heatmap_image = Image.fromarray(heatmap_array).resize(image.size)
    heatmap_color = np.uint8(cm.jet(np.asarray(heatmap_image))[:, :, :3] * 255)
    colored_heatmap = Image.fromarray(heatmap_color).convert("RGB")
    return Image.blend(image.convert("RGB"), colored_heatmap, opacity)