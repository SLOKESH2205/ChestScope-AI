"""
Explainable AI (XAI) module for Chest X-Ray Disease Classification.

Provides Grad-CAM, Grad-CAM++, and Integrated Gradients implementations
to visualize regions of input images that influence prediction decisions.
Supports both Sequential and Functional Keras model architectures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# Standard image size
IMG_SIZE = (224, 224)

def get_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        # Look for conv layer within sequential model or nested base models
        if isinstance(layer, keras.Model) or hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if 'conv' in sub_layer.name.lower():
                    return f"{layer.name}/{sub_layer.name}"
        if 'conv' in layer.name.lower():
            return layer.name
    raise ValueError("Could not find a convolutional layer in the model.")

def get_sub_model_and_layer(model: keras.Model, target_layer_path: str) -> Tuple[keras.Model, keras.layers.Layer]:
    """Resolve sub-model and layer reference for nested models (like transfer learning base models)."""
    if '/' in target_layer_path:
        base_name, layer_name = target_layer_path.split('/')
        base_model = model.get_layer(base_name)
        target_layer = base_model.get_layer(layer_name)
        return base_model, target_layer
    else:
        target_layer = model.get_layer(target_layer_path)
        return model, target_layer

def compute_gradcam(
    model: keras.Model,
    image: np.ndarray,
    pred_index: int = None,
    layer_name: str = None
) -> np.ndarray:
    """
    Compute standard Grad-CAM heatmap.
    """
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        
    is_sequential = isinstance(model, keras.Sequential) or model.__class__.__name__ == 'Sequential'
    
    if is_sequential and '/' not in layer_name:
        conv_idx = -1
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_name:
                conv_idx = idx
                break
                
        if conv_idx == -1:
            logger.warning(f"Could not find layer {layer_name} in Sequential model.")
            return np.zeros(IMG_SIZE)
            
        with tf.GradientTape() as tape:
            x = tf.convert_to_tensor(image, dtype=tf.float32)
            conv_outputs = None
            for idx, layer in enumerate(model.layers):
                x = layer(x)
                if idx == conv_idx:
                    conv_outputs = x
                    tape.watch(conv_outputs)
            predictions = x
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Functional model
        if '/' in layer_name:
            base_name, sub_layer_name = layer_name.split('/')
            base_model = model.get_layer(base_name)
            
            grad_model = keras.Model(
                inputs=base_model.inputs,
                outputs=[base_model.get_layer(sub_layer_name).output, base_model.outputs[0]]
            )
            
            base_input = image
            for layer in model.layers:
                if layer.name == base_name:
                    break
                base_input = layer(base_input)
                
            with tf.GradientTape() as tape:
                conv_outputs, base_features = grad_model(base_input)
                head_input = base_features
                passed_base = False
                for layer in model.layers:
                    if passed_base:
                        head_input = layer(head_input)
                    if layer.name == base_name:
                        passed_base = True
                predictions = head_input
                
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
                
            grads = tape.gradient(class_channel, conv_outputs)
        else:
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer_name).output, model.outputs[0]]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
                
            grads = tape.gradient(class_channel, conv_outputs)
            
    if grads is None:
        logger.warning(f"Grad-CAM gradients are None for layer {layer_name}")
        return np.zeros(IMG_SIZE)
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.nn.relu(heatmap)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom
    
    # Resize to original input size
    heatmap = tf.image.resize(tf.expand_dims(heatmap, axis=-1), IMG_SIZE)
    return tf.squeeze(heatmap).numpy()

def compute_gradcam_plusplus(
    model: keras.Model,
    image: np.ndarray,
    pred_index: int = None,
    layer_name: str = None
) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap for fine-grained localization.
    """
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        
    is_sequential = isinstance(model, keras.Sequential) or model.__class__.__name__ == 'Sequential'
    
    if is_sequential and '/' not in layer_name:
        conv_idx = -1
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_name:
                conv_idx = idx
                break
                
        if conv_idx == -1:
            logger.warning(f"Could not find layer {layer_name} in Sequential model.")
            return np.zeros(IMG_SIZE)
            
        with tf.GradientTape() as tape_val:
            with tf.GradientTape() as tape_add:
                with tf.GradientTape() as tape_grads:
                    x = tf.convert_to_tensor(image, dtype=tf.float32)
                    conv_outputs = None
                    for idx, layer in enumerate(model.layers):
                        x = layer(x)
                        if idx == conv_idx:
                            conv_outputs = x
                            tape_grads.watch(conv_outputs)
                            tape_add.watch(conv_outputs)
                            tape_val.watch(conv_outputs)
                    predictions = x
                    if pred_index is None:
                        pred_index = tf.argmax(predictions[0])
                    class_channel = predictions[:, pred_index]
                    
                grads = tape_grads.gradient(class_channel, conv_outputs)
            grads_squared = tape_add.gradient(grads, conv_outputs)
        grads_cubed = tape_val.gradient(grads_squared, conv_outputs)
    else:
        # Functional model
        if '/' in layer_name:
            base_name, sub_layer_name = layer_name.split('/')
            base_model = model.get_layer(base_name)
            grad_model = keras.Model(
                inputs=base_model.inputs,
                outputs=[base_model.get_layer(sub_layer_name).output, base_model.outputs[0]]
            )
            base_input = image
            for layer in model.layers:
                if layer.name == base_name:
                    break
                base_input = layer(base_input)
        else:
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer_name).output, model.outputs[0]]
            )
            base_input = image
            base_name = None
            
        with tf.GradientTape() as tape_val:
            with tf.GradientTape() as tape_add:
                with tf.GradientTape() as tape_grads:
                    conv_outputs, base_features = grad_model(base_input)
                    if base_name:
                        head_input = base_features
                        passed_base = False
                        for layer in model.layers:
                            if passed_base:
                                head_input = layer(head_input)
                            if layer.name == base_name:
                                passed_base = True
                        predictions = head_input
                    else:
                        predictions = base_features
                        
                    if pred_index is None:
                        pred_index = tf.argmax(predictions[0])
                    class_channel = predictions[:, pred_index]
                    
                grads = tape_grads.gradient(class_channel, conv_outputs)
            grads_squared = tape_add.gradient(grads, conv_outputs)
        grads_cubed = tape_val.gradient(grads_squared, conv_outputs)
        
    if grads is None or grads_squared is None or grads_cubed is None:
        logger.warning(f"Grad-CAM++ calculations failed for layer {layer_name}")
        return np.zeros(IMG_SIZE)
        
    # Convert to numpy arrays for calculation
    grads = grads.numpy()[0]
    grads_squared = grads_squared.numpy()[0]
    grads_cubed = grads_cubed.numpy()[0]
    conv_outputs = conv_outputs.numpy()[0]
    
    # Calculate alpha weights for Grad-CAM++
    global_sum = np.sum(conv_outputs, axis=(0, 1))
    
    alpha_denom = 2.0 * grads_squared + conv_outputs * grads_cubed
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-8)
    
    alphas = grads_squared / alpha_denom
    
    weights = np.maximum(grads, 0.0)
    alphas_guided = alphas * weights
    
    guided_weights = np.sum(alphas_guided, axis=(0, 1))
    
    heatmap = np.sum(guided_weights * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0.0)
    denom = np.max(heatmap) + 1e-8
    heatmap = heatmap / denom
    
    # Resize to original input size
    heatmap = tf.image.resize(tf.expand_dims(heatmap, axis=-1), IMG_SIZE)
    return tf.squeeze(heatmap).numpy()

def compute_integrated_gradients(
    model: keras.Model,
    image: np.ndarray,
    pred_index: int = None,
    num_steps: int = 50
) -> np.ndarray:
    """
    Compute Integrated Gradients mapping for input attribution.
    """
    if len(image.shape) == 4:
        image = image[0]
        
    # Baseline is all zeros (black image)
    baseline = np.zeros(image.shape, dtype=np.float32)
    
    # Generate interpolated images along path [baseline -> image]
    alphas = np.linspace(0.0, 1.0, num_steps + 1)
    interpolated_images = [baseline + alpha * (image - baseline) for alpha in alphas]
    interpolated_images = np.array(interpolated_images, dtype=np.float32)
    
    # Predict and compute gradients for each interpolated step
    grads = []
    
    # Process in smaller sub-batches to prevent memory overflow
    batch_size = 10
    for i in range(0, len(interpolated_images), batch_size):
        sub_batch = interpolated_images[i:i+batch_size]
        sub_batch = tf.convert_to_tensor(sub_batch)
        
        with tf.GradientTape() as tape:
            tape.watch(sub_batch)
            predictions = model(sub_batch)
            if pred_index is None:
                # Use target prediction of the original image
                orig_pred = model(np.expand_dims(image, axis=0))
                pred_index = tf.argmax(orig_pred[0]).numpy()
            class_channel = predictions[:, pred_index]
            
        sub_grads = tape.gradient(class_channel, sub_batch)
        grads.append(sub_grads.numpy())
        
    grads = np.vstack(grads)
    
    # Approximate integral using Riemann sum (trapezoidal rule)
    avg_grads = np.mean(grads[:-1] + grads[1:], axis=0) / 2.0
    
    # IG = (image - baseline) * avg_grads
    integrated_grads = (image - baseline) * avg_grads
    
    # Sum over channels and normalize
    attribution = np.sum(np.abs(integrated_grads), axis=-1)
    denom = np.max(attribution) + 1e-8
    attribution = attribution / denom
    
    return attribution

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """Blend original image with heatmap representation."""
    if len(image.shape) == 4:
        image = image[0]
        
    # Force float [0, 1] scaling
    if image.max() > 1.0:
        image = image / 255.0
        
    try:
        from matplotlib import colormaps
        cmap = colormaps[colormap]
    except ImportError:
        cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]
    
    blended = alpha * heatmap_colored + (1.0 - alpha) * image
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)

def generate_xai_plots(
    model: keras.Model,
    image: np.ndarray,
    pred_index: int,
    output_dir: Path | str,
    prefix: str = ""
) -> Dict[str, str]:
    """
    Generate and save Grad-CAM, Grad-CAM++, and Integrated Gradients visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute heatmaps
    gradcam_map = compute_gradcam(model, image, pred_index)
    gradcam_pp_map = compute_gradcam_plusplus(model, image, pred_index)
    ig_map = compute_integrated_gradients(model, image, pred_index)
    
    # Create overlays
    gc_overlay = overlay_heatmap(image, gradcam_map)
    gc_pp_overlay = overlay_heatmap(image, gradcam_pp_map)
    
    # Save composite plots
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # Original image
    orig_img = image[0] if len(image.shape) == 4 else image
    if orig_img.max() > 1.0:
        orig_img = orig_img / 255.0
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Chest X-Ray")
    axes[0].axis('off')
    
    # Grad-CAM
    axes[1].imshow(gc_overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis('off')
    
    # Grad-CAM++
    axes[2].imshow(gc_pp_overlay)
    axes[2].set_title("Grad-CAM++")
    axes[2].axis('off')
    
    # Integrated Gradients
    axes[3].imshow(ig_map, cmap='hot')
    axes[3].set_title("Integrated Gradients")
    axes[3].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / f"{prefix}xai_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # Save standalone Grad-CAM example as required
    fig_gc, ax_gc = plt.subplots(figsize=(6, 6))
    ax_gc.imshow(gc_overlay)
    ax_gc.axis('off')
    ax_gc.set_title("Grad-CAM Highlighted Lung Lesions")
    gc_example_path = output_dir / f"{prefix}gradcam_example.png"
    fig_gc.savefig(gc_example_path, dpi=200, bbox_inches='tight')
    plt.close(fig_gc)
    
    return {
        "xai_comparison": str(plot_path),
        "gradcam_example": str(gc_example_path)
    }
