import argparse
import datetime
import os
import threading

import cv2
from flask import Flask, Response
from flask import render_template

from src.main_face_recognition import wide_image_generator

class App(Flask):
    def __init__(self, name, template_folder=None):
        super().__init__(name, template_folder=template_folder, static_folder=template_folder)
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        self.add_url_rules()
        self.updater_thread = CurrentFrameUpdaterThread(self)

    def add_url_rules(self):
        self.add_url_rule("/", "index", self.index)
        self.add_url_rule("/video_feed", "video_feed", self.video_feed)
        self.add_url_rule("/bootstrap.min.css", "bootstrap_min_css", self.bootstrap_min_css)

    def index(self):
        return render_template("index.html")

    def bootstrap_min_css(self):
        return render_template("bootstrap.min.css")

    def video_feed(self):
        return Response(self.current_frame_generator(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    def current_frame_generator(self):
        while True:
            with self.current_frame_lock:
                if self.current_frame is None:
                    continue
                try:
                    (flag, encodedImage) = cv2.imencode(".jpg", self.current_frame)
                except:
                    print(self.current_frame)
                    raise
                if not flag:
                    continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')


class CurrentFrameUpdaterThread(threading.Thread):
    def __init__(self, app: App):
        super().__init__(target=self.main_loop, daemon=True)
        self.app = app
        self.start()

    def main_loop(self):
        kill_flag = threading.Event()

        for frame in wide_image_generator(kill_flag, "data/target_face_images/", "weights/yolov8n-face.trt",
                                          "weights/arcfaceresnet100-8.trt", 0.65, 0.5, 0.5):
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            with self.app.current_frame_lock:
                self.app.current_frame = frame.copy()


def main(ip, port):
    print(os.getcwd())
    app = App(__name__, template_folder="./templates/")
    app.run(host=ip, port=port, debug=True,
            threaded=True, use_reloader=False)


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--ip", type=str, required=True)
    args_parser.add_argument("--port", type=int, required=True)
    args = args_parser.parse_args()
    main(args.ip, args.port)
