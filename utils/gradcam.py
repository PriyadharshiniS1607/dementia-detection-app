import numpy as np
import cv2
import tensorflow as tf


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    """
    Generate Grad-CAM heatmap for a single image.
    img_array shape: (1, 224, 224, 3)
    """

    # Get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Create model that maps input -> last conv output + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Compute gradient of top predicted class with respect to conv output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Safety: if predictions comes as list, take first tensor
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Safety: ensure tensor
        predictions = tf.convert_to_tensor(predictions)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    # Gradient of class score with respect to feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Average gradients over width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight feature maps by importance
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    original_img should be OpenCV BGR image.
    """

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply color map
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay
    alpha = 0.6
    overlay = cv2.addWeighted(original_img, 1 - alpha, color_heatmap, alpha, 0)

    return color_heatmap, overlay