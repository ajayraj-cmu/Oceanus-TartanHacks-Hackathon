import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import subprocess
import logging
import torch
from transformers import pipeline
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

# --------------------------------------------------
# Optimization settings for CPU-based processing
# --------------------------------------------------
cv2.setNumThreads(0)  # Prevent OpenCV from oversubscribing threads.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(4)  # Limit PyTorch threads on CPU.

# ---------------------------
# Global Configuration & Logging
# ---------------------------
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ---------------------------
# Model Initialization
# ---------------------------
# YOLOv8 (nano version)
yolo_model = YOLO('yolov8n.pt')

# Depth estimation pipeline.
depth_pipe = pipeline("depth-estimation",
                      model="depth-anything/Depth-Anything-V2-Small-hf",
                      device=device)

# SAM (Segment Anything) for ground segmentation.
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure this file is present.
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# ---------------------------
# Asynchronous Inference Setup & Constants
# ---------------------------
downscale_factor = 0.5      # Process heavy models on a smaller copy.
HEAVY_UPDATE_INTERVAL = 5   # Run heavy inference every 5 frames.
YOLO_UPDATE_INTERVAL = 2    # Run YOLO detection every 2 frames.
frame_count = 0

# Cached heavy outputs.
last_depth_array = None       # Full-resolution depth map.
last_ground_mask = None       # Full-resolution SAM ground mask.
inference_future = None        # Future for heavy inference.

# Cached YOLO outputs.
yolo_future = None             # Future for YOLO detection.
last_yolo_results = None       # Cached YOLO results.

# Use a thread pool with 2 workers.
executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------
# Aesthetic Helper Function
# ---------------------------
def draw_label(img, text, pos, font=cv2.FONT_HERSHEY_COMPLEX, scale=0.6, thickness=2, 
               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draws text on an image with a filled background rectangle for better readability.
    """
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - text_size[1] - baseline), (x + text_size[0], y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

# ---------------------------
# Utility Functions
# ---------------------------
def speak_mac(text):
    """Speak text using macOS's 'say' command."""
    sanitized_text = text.replace("'", "\\'")
    try:
        subprocess.run(["say", sanitized_text], check=True)
    except subprocess.CalledProcessError as e:
        print("speak_mac error:", e)

def heavy_inference(small_frame, full_width, full_height):
    """
    Run depth estimation and SAM segmentation on a downscaled frame.
    A single BGR→RGB conversion is reused for both models.
    """
    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # --- Depth Estimation ---
    pil_image = Image.fromarray(small_frame_rgb)
    depth_result = depth_pipe(pil_image)
    depth_small = np.array(depth_result["depth"])
    depth_array = cv2.resize(depth_small, (full_width, full_height))
    
    # --- SAM Segmentation ---
    predictor.set_image(small_frame_rgb)
    # Use a bounding box covering the lower 40% of the image.
    box_prompt = np.array([0, int(small_frame.shape[0] * 0.6),
                           small_frame.shape[1], small_frame.shape[0]])
    masks, scores, _ = predictor.predict(box=box_prompt, multimask_output=True)
    best_index = int(np.argmax(scores))
    ground_mask_small = (masks[best_index].astype(np.uint8)) * 255
    ground_mask = cv2.resize(ground_mask_small, (full_width, full_height),
                             interpolation=cv2.INTER_NEAREST)
    return depth_array, ground_mask

def depth_heatmap(depth_array, frame):
    """
    Generate a colorized depth heat map from the depth array.
    Uses the TURBO colormap for a bright, modern look.
    Returns a black image if depth_array is None.
    """
    if depth_array is not None:
        norm_depth = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        norm_depth_uint8 = norm_depth.astype(np.uint8)
        # Use TURBO colormap instead of MAGMA.
        return cv2.applyColorMap(norm_depth_uint8, cv2.COLORMAP_TURBO)
    return np.zeros_like(frame)

def run_yolo_detection(frame, depth_array):
    """
    Run YOLO detection on the given frame.
    Returns:
      - detection_data: list of (label, x1, y1, x2, y2, avg_depth)
      - obstacle_mask: binary mask of detected obstacles
      - frame_with_boxes: camera view with YOLO bounding boxes drawn in magenta
      - depth_overlay: depth heat map with YOLO boxes and labels overlaid
    """
    height, width, _ = frame.shape
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)
    detection_data = []
    # Create copies to draw on.
    frame_with_boxes = frame.copy()
    depth_overlay = depth_heatmap(depth_array, frame)
    
    results = yolo_model(frame)
    for box in results[0].boxes:
        coords = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
        conf = box.conf.tolist()[0]
        cls = int(box.cls.tolist()[0])
        x1, y1, x2, y2 = map(int, coords)
        label = (yolo_model.names.get(cls, str(cls))
                 if hasattr(yolo_model, 'names') and isinstance(yolo_model.names, dict)
                 else str(cls))
        # Draw a magenta bounding box.
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.rectangle(depth_overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
        if depth_array is not None:
            region = depth_array[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]
            avg_depth = np.mean(region) if region.size > 0 else 0.0
        else:
            avg_depth = 0.0
        text = f"{label}: {conf:.2f}, D:{avg_depth:.2f}"
        # Draw text with our custom label function.
        draw_label(frame_with_boxes, text, (x1, y1 - 5),
                   font=cv2.FONT_HERSHEY_COMPLEX, scale=0.6, thickness=2,
                   text_color=(255,255,255), bg_color=(0,0,0))
        draw_label(depth_overlay, text, (x1, y1 - 5),
                   font=cv2.FONT_HERSHEY_COMPLEX, scale=0.6, thickness=2,
                   text_color=(255,255,255), bg_color=(0,0,0))
        cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), 255, -1)
        detection_data.append((label, x1, y1, x2, y2, avg_depth))
    return detection_data, obstacle_mask, frame_with_boxes, depth_overlay

def dummy_pathfinder(data, target_obj):
    """Dummy path planning function."""
    return "move forward", "middle of screen"

def get_object_mask(frame, prompt):
    """Dummy object mask function (placeholder)."""
    return np.zeros(frame.shape[:2], dtype=np.uint8)

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    global frame_count, inference_future, yolo_future
    global last_depth_array, last_ground_mask, last_yolo_results

    target_obj = "bottle"
    speak_mac(f"Perfect! Let's go find your '{target_obj}'")
    TIME_SPEAK_INTERVAL = 0.75

    # Instead of using the webcam, open an MP4 (or MOV) file.
    video_file = "video-160_singular_display.mov"  # <-- Change this to your video file path.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    interval_time = 0
    last_frame = None  # For simple motion detection.
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        # If we have reached the end of the video, reset to the beginning.
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        # For speed, work with a lower resolution.
        frame = cv2.resize(frame, (480, 360))
        height, width, _ = frame.shape

        # Simple motion check: if frame differences are very small, skip heavy inference.
        if last_frame is not None:
            diff = cv2.absdiff(frame, last_frame)
            nonzero = np.count_nonzero(diff)
            skip_heavy = nonzero < 100  # Adjust threshold as needed.
        else:
            skip_heavy = False
        last_frame = frame.copy()

        # Create a downscaled copy for heavy inference.
        small_frame = cv2.resize(frame, (int(width * downscale_factor), int(height * downscale_factor)))
        
        # --- Asynchronous Heavy Inference (Depth & SAM) ---
        if not skip_heavy and frame_count % HEAVY_UPDATE_INTERVAL == 0:
            if inference_future is None or inference_future.done():
                inference_future = executor.submit(heavy_inference, small_frame, width, height)
        if inference_future is not None and inference_future.done():
            try:
                last_depth_array, last_ground_mask = inference_future.result()
            except Exception as e:
                print("Error in heavy inference:", e)
            inference_future = None

        # --- Asynchronous YOLO Detection ---
        if frame_count % YOLO_UPDATE_INTERVAL == 0:
            if yolo_future is None or yolo_future.done():
                yolo_future = executor.submit(run_yolo_detection, frame, last_depth_array)
        if yolo_future is not None and yolo_future.done():
            try:
                last_yolo_results = yolo_future.result()
            except Exception as e:
                print("Error in YOLO detection:", e)
                last_yolo_results = None

        # Use the latest YOLO results if available; otherwise, run synchronously.
        if last_yolo_results is not None:
            detection_data, obstacle_mask, frame_with_boxes, depth_overlay = last_yolo_results
        else:
            detection_data, obstacle_mask, frame_with_boxes, depth_overlay = run_yolo_detection(frame, last_depth_array)
        
        # --- Compute a "tight" SAM ground mask ---
        # Subtract YOLO-detected obstacles from the SAM segmentation.
        if last_ground_mask is not None:
            open_ground_mask = cv2.bitwise_and(last_ground_mask, cv2.bitwise_not(obstacle_mask))
            # Apply erosion to "tighten" the mask around objects.
            kernel = np.ones((5, 5), np.uint8)
            tight_sam_mask = cv2.erode(open_ground_mask, kernel, iterations=1)
        else:
            tight_sam_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert the binary SAM mask to a colored overlay.
        colored_sam_mask = np.zeros((height, width, 3), dtype=np.uint8)
        # Here we choose a light purple color for the free-to-roam areas.
        colored_sam_mask[tight_sam_mask == 255] = (180, 105, 255)

        # ---------------------------
        # Display Windows:
        # 1. YOLO Bounding Boxes over camera view.
        cv2.imshow("YOLO Bounding Boxes", frame_with_boxes)
        # 2. Depth Model with YOLO tags over the depth heatmap.
        cv2.imshow("Depth Model with YOLO", depth_overlay)
        # 3. Tight SAM Ground Mask (colored overlay).
        cv2.imshow("Tight SAM Ground Mask", colored_sam_mask)
        # 4. Plain Camera View.
        cv2.imshow("Camera View", frame)
        
        # Dummy path planning and periodic speech.
        move, location = dummy_pathfinder(detection_data, target_obj)
        interval_time += time.time() - start_time
        if interval_time > TIME_SPEAK_INTERVAL:
            speak_mac(move)
            interval_time = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()
