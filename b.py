# import the necessary packages
import argparse
import datetime
import multiprocessing
import threading
import time

import cv2
import imutils
from flask import Flask, Response
from flask import render_template
from imutils.video import VideoStream

from src.main import run

# initialize a flask object
app = Flask(__name__, template_folder="./")


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion(reader):
    try:
        multiprocessing.set_start_method("spawn")
    except:
        print("Not using start method spawn, probably on windows.")

    kill_flag = multiprocessing.Event()

    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    # loop over frames from the video stream
    for frame in run(kill_flag):
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        # frame = reader.read()
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def main(ip, port):
    global outputFrame, lock
    outputFrame = None
    lock = threading.Lock()

    vs = None # VideoStream(src=0).start()
    time.sleep(2.0)

    t = threading.Thread(target=detect_motion, args=(
        vs,))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=ip, port=port, debug=True,
            threaded=True, use_reloader=False)

    vs.stop()


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
