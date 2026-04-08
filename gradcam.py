import torch
import cv2
import numpy as np
from PIL import Image

# Hook storage
gradients = None
activations = None

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

def save_activations(module, input, output):
    global activations
    activations = output.detach()

def _ensure_pil_image(image_source):
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    return Image.open(image_source).convert("RGB")

def generate_gradcam(model, image_source, transform, device):
    global gradients, activations

    model.eval()

    # Load image
    image = _ensure_pil_image(image_source)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Hook last conv layer (ResNet layer4)
    target_layer = model.layer4
    forward_handle = target_layer.register_forward_hook(save_activations)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    # Forward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    forward_handle.remove()
    backward_handle.remove()

    # Convert to numpy
    grads = gradients.cpu().numpy()[0]
    acts = activations.cpu().numpy()[0]

    # Global average pooling
    weights = np.mean(grads, axis=(1, 2))

    # Weighted sum
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize
    cam = cam - np.min(cam)
    max_cam = np.max(cam)
    if max_cam > 0:
        cam = cam / max_cam

    # Resize
    cam = cv2.resize(cam, (224, 224))

    return cam, image

def overlay_heatmap(cam, image):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    image = np.array(image.resize((224, 224)))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = heatmap * 0.4 + image_bgr

    return np.uint8(overlay)
