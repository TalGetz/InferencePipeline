import argparse
import datetime
import multiprocessing
import threading
import time

import cv2
from flask import Flask, Response
from flask import render_template

from src.main_face_recognition import wide_image_generator

app = Flask(__name__, template_folder="./")


@app.route("/")
def index():
    return render_template("index.html")


def detect_motion():
    kill_flag = multiprocessing.Event()

    global outputFrame, lock

    for frame in wide_image_generator(kill_flag, "data/target_face_images/", "weights/yolov8n-face.trt",
                                      "weights/arcfaceresnet100-8.trt", 0.65, 0.5, 0.5):
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        with lock:
            outputFrame = frame.copy()


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue

            outputFrame = cv2.resize(outputFrame, (800, 800))
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def main(ip, port):
    global outputFrame, lock
    outputFrame = None
    lock = threading.Lock()

    time.sleep(2.0)

    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()
    app.run(host=ip, port=port, debug=True,
            threaded=True, use_reloader=False)


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = ap.parse_args()
    main(args.ip, args.port)
