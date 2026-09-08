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
    base_model = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model)),
        None,
    )
    if base_model is None:
        raise ValueError("لم يتم العثور على نموذج VGG16 داخل النموذج")

    conv_layer_index = base_model.layers.index(last_conv_layer)
    prediction_tensor = last_conv_layer.output
    for layer in base_model.layers[conv_layer_index + 1:]:
        prediction_tensor = layer(prediction_tensor)
    base_model_index = model.layers.index(base_model)
    for layer in model.layers[base_model_index + 1:]:
        prediction_tensor = layer(prediction_tensor)

    gradient_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, prediction_tensor],
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


def heatmap_to_image(heatmap, size):
    """Convert a normalized Grad-CAM array into a colored PIL image."""
    heatmap_array = np.uint8(255 * heatmap)
    heatmap_image = Image.fromarray(heatmap_array).resize(size)
    heatmap_color = np.uint8(cm.jet(np.asarray(heatmap_image))[:, :, :3] * 255)
    return Image.fromarray(heatmap_color).convert("RGB")


def overlay_heatmap(image, heatmap, opacity=0.4):
    """Overlay a colored heatmap on the original PIL image."""
    colored_heatmap = heatmap_to_image(heatmap, image.size)
    return Image.blend(image.convert("RGB"), colored_heatmap, opacity)