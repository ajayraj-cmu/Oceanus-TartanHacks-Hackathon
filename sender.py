from flask import Flask, Response
import cv2
import mss
import numpy as np

app = Flask(__name__)
screen_capture = mss.mss()
region = {"top": 100, "left": 100, "width": 640, "height": 480}  # Adjust as needed

def generate_frames():
    while True:
        frame = screen_capture.grab(region)
        img = np.array(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
