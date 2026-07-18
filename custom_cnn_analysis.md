# Engineering Report: Custom CNN Architecture Analysis

An investigation into the parameter distribution, optimizer variables, memory footprint, and architectural optimizations of the baseline Custom CNN model.

---

## 📊 Layer-by-Layer Architectural Breakdown

The model loaded from `custom_cnn.h5` contains a 3-block convolutional feature extractor followed by a dense classifier head. Below is the layer-by-layer breakdown:

| Layer Name | Type | Input Shape | Output Shape | Kernel Shape | Parameter Formula | Param Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `conv2d` | Conv2D | (224, 224, 3) | (222, 222, 32) | (3, 3, 3, 32) | `(3*3*3 * 32) + 32` | 896 |
| `max_pooling2d` | MaxPooling2D | (222, 222, 32) | (111, 111, 32) | - | - | 0 |
| `conv2d_1` | Conv2D | (111, 111, 32) | (109, 109, 64) | (3, 3, 32, 64) | `(3*3*32 * 64) + 64` | 18,496 |
| `max_pooling2d_1`| MaxPooling2D | (109, 109, 64) | (54, 54, 64) | - | - | 0 |
| `conv2d_2` | Conv2D | (54, 54, 64) | (52, 52, 128) | (3, 3, 64, 128) | `(3*3*64 * 128) + 128` | 73,856 |
| `max_pooling2d_2`| MaxPooling2D | (52, 52, 128) | (26, 26, 128) | - | - | 0 |
| `flatten` | Flatten | (26, 26, 128) | (86528,) | - | - | 0 |
| `dense` | Dense | (86528,) | (512,) | (86528, 512) | `(86528 * 512) + 512` | **44,302,848** |
| `dropout` | Dropout | (512,) | (512,) | - | - | 0 |
| `dense_1` | Dense | (512,) | (4,) | (512, 4) | `(512 * 4) + 4` | 2,052 |

*   **Total Trainable Parameters:** **44,398,148**
*   **Total Non-trainable Parameters:** 0

---

## 🔍 Why is Model Size ~508 MB (133M variables)?

When inspecting variables saved inside the legacy H5 checkpoint, the parameter count is reported as approximately **133.2 million variables**, resulting in a **508 MB** file size on disk. 

This discrepency is caused by **Optimizer State Variable Serialization**:
*   The model was trained using the **Adam** optimizer.
*   Adam tracks **two running moments** (momentum \(m_t\) and velocity \(v_t\)) for *every single trainable parameter* to compute adaptive learning rates.
*   Therefore, the saved H5 file stores:
    1.  **Model Weights:** $44,398,148$ parameters
    2.  **Momentum States ($m_t$):** $44,398,148$ variables
    3.  **Velocity States ($v_t$):** $44,398,148$ variables
*   **Total Serialized Variables:** $3 \times 44,398,148 = \mathbf{133,194,444}$ variables.
*   At $4$ bytes per single-precision float32, this takes $133,194,444 \times 4 \text{ bytes} \approx \mathbf{532.8 \text{ MB}}$ of raw storage (compressed to $508 \text{ MB}$ on disk).

---

## 💡 Optimization Proposal: GlobalAveragePooling2D

The model's parameter bottleneck is the transition from the last convolutional feature map to the dense layer:
*   **Flattening** maps a $(26, 26, 128)$ activation map directly to a vector of shape $(86528,)$. This results in **44,302,848** parameters in the `dense` layer ($99.78\%$ of the entire model!).

### Optimization Design:
Replace `Flatten` with `GlobalAveragePooling2D`. Instead of flattening the spatial dimensions, we take the average activation of each of the $128$ feature channels over the $26 \times 26$ grid:
1.  Output of feature extractor: $(None, 26, 26, 128)$
2.  Output of `GlobalAveragePooling2D`: $(None, 128)$
3.  New input to `dense` layer: $128$ features instead of $86528$.

### Param comparison:
*   **Flatten `dense` weight count:** $86,528 \times 512 = 44,302,848$
*   **Global Average Pooling `dense` weight count:** $128 \times 512 = \mathbf{65,536}$
*   **Net Parameter Reduction:** $\mathbf{44,237,312}$ parameters saved ($99.64\%$ total parameter reduction!).
*   **Disk Size Impact:** Model H5 file size would drop from **508 MB** to **less than 1.5 MB**!

### Impact on Validation Performance:
*   **Overfitting Reduction:** A dense layer with 44M parameters easily overfits on a small dataset of 532 training images. Replacing it with GAP forces translation invariance and regularizes the model, which typically **improves validation accuracy** on medical imaging cohorts.
