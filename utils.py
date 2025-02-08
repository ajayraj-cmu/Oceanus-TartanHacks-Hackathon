#logic inspired by prior project
import cv2
import numpy as np
from PIL import Image
import subprocess
import sounddevice as sd
import time
import torch
import pyttsx3
import scipy.signal
from transformers import pipeline, CLIPSegForImageSegmentation, CLIPSegProcessor

# Initialize models
asr_pipeline = pipeline("automatic-speech-recognition")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")


def speak_mac(text):
    """Speak text on macOS using the 'say' command or fallback to pyttsx3 for non-macOS."""
    sanitized_text = text.replace("'", "\\'")
    try:
        subprocess.run(["say", sanitized_text], check=True)
    except FileNotFoundError:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("speak_mac error:", e)


def record_audio(record_duration=3, sample_rate=16000):
    """Record audio and apply optional noise reduction."""
    try:
        print("Recording... Please speak.")
        audio = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        print("Recording complete.")
        # Apply a simple high-pass filter to reduce noise
        b, a = scipy.signal.butter(4, 100 / (sample_rate / 2), btype='high')
        filtered_audio = scipy.signal.filtfilt(b, a, audio.flatten())
        return filtered_audio
    except Exception as e:
        print("Audio recording failed:", e)
        return np.zeros(int(record_duration * sample_rate))


def get_target_object(record_duration=3):
    """Use audio recording and Whisper ASR to extract the target object."""
    audio_data = record_audio(record_duration)
    try:
        # Convert to 16-bit PCM WAV for ASR
        audio_float32 = (audio_data * 32767).astype(np.int16).tobytes()
        transcription = asr_pipeline(audio_float32, sampling_rate=16000)
        transcribed_text = transcription['text']
        print("Transcribed text:", transcribed_text)
        import re
        match = re.search(r'help me find my (.+)', transcribed_text.lower())
        if match:
            target = match.group(1).strip()
            print("Target object detected:", target)
            return target
    except Exception as e:
        print("ASR error:", e)
    return None


def get_object_mask(frame, prompt):
    """Use CLIPSeg for object segmentation based on a text prompt."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clipseg_processor(text=prompt, images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
        mask = outputs.logits.sigmoid().squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def run_depth_heatmap(depth_array, frame):
    """Generate a colorized depth heat map from a depth array with proper scaling."""
    if depth_array is not None:
        norm_depth = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        norm_depth_uint8 = norm_depth.astype(np.uint8)
        depth_colored = cv2.applyColorMap(norm_depth_uint8, cv2.COLORMAP_MAGMA)
        blended = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)
        return blended
    return np.zeros_like(frame)


def run_yolo_detection(frame, depth_array, yolo_model):
    """Run YOLO detection on the frame and overlay bounding boxes with depth information."""
    start_time = time.time()
    detection_data = []
    height, width, _ = frame.shape
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)
    frame_with_boxes = frame.copy()
    depth_overlay = run_depth_heatmap(depth_array, frame)

    results = yolo_model(frame)
    for result in results[0].boxes:
        box = result.xyxy.tolist()[0]  # [x1, y1, x2, y2]
        conf = result.conf.tolist()[0]
        cls = int(result.cls.tolist()[0])
        x1, y1, x2, y2 = map(int, box)
        label = yolo_model.names[cls] if yolo_model.names and cls in yolo_model.names else str(cls)

        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if depth_array is not None:
            region = depth_array[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]
            avg_depth = np.mean(region) if region.size > 0 else 0.0
        else:
            avg_depth = 0.0
        text = f"{label}: {conf:.2f}, D:{avg_depth:.2f}m"
        cv2.putText(frame_with_boxes, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detection_data.append((label, x1, y1, x2, y2, avg_depth))

    fps = 1 / (time.time() - start_time)
    print(f"YOLO Detection FPS: {fps:.2f}")
    return detection_data, obstacle_mask, detection_data, frame_with_boxes, depth_overlay


def heavy_inference(small_frame, full_width, full_height, depth_pipe, predictor):
    """Perform heavy inference (depth estimation and SAM segmentation) on a downscaled frame."""
    pil_image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
    depth_result = depth_pipe(pil_image)
    depth_small = np.array(depth_result["depth"])
    depth_array = cv2.resize(depth_small, (full_width, full_height))
    
    predictor.set_image(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
    box_prompt = np.array([0, int(small_frame.shape[0] * 0.6), small_frame.shape[1], small_frame.shape[0]])
    masks, scores, logits = predictor.predict(box=box_prompt, multimask_output=True)
    best_index = int(np.argmax(scores))
    ground_mask_small = masks[best_index]
    ground_mask_small = (ground_mask_small.astype(np.uint8)) * 255
    ground_mask = cv2.resize(ground_mask_small, (full_width, full_height), interpolation=cv2.INTER_NEAREST)
    return depth_array, ground_mask
