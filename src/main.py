import argparse
import multiprocessing
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import tqdm

from src import config
from src.frame_readers.camera_reader_process import CameraReaderProcess
from src.models.detection.face_detection.yolov8nface.yolov8nface import YOLOv8nFace
from src.models.recognition.face_recognition.arcfaceresnet100.arcfaceresnet100 import ArcFaceResnet100
from src.models.recognition.face_recognition.arcfaceresnet100.item import ArcFaceResnet100Item
from src.models.recognition.face_recognition.arcfaceresnet100.target import ArcFaceResnet100Target


def main():
    try:
        multiprocessing.set_start_method("spawn")
    except:
        print("Not using start method spawn, probably on windows.")

    kill_flag = threading.Event()
    if config.DEBUG:
        run(kill_flag)
    else:
        try:
            run(kill_flag)
        except:
            pass

    kill_flag.set()
    print("shutting down: main")
    time.sleep(3)


def run(kill_flag):
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.trt',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.65, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument("--targetsFolderPath", default="data/target_face_images/", type=str)
    parser.add_argument("--ip", default="data/target_face_images/", type=str)
    parser.add_argument("--port", default=1, type=int)
    args = parser.parse_args()

    camera_reader_process = CameraReaderProcess(kill_flag=kill_flag).start()
    yolov8nface_main_thread = YOLOv8nFace(0, args.modelpath, conf_threshold=args.confThreshold,
                                          iou_threshold=args.nmsThreshold)

    target_file_names = os.listdir(args.targetsFolderPath)
    target_names = [x[:-4] for x in target_file_names if x.endswith(".jpg") or x.endswith(".png")]
    target_images = [cv2.imread(str(Path(args.targetsFolderPath) / file_name)) for file_name in target_file_names]
    targets = []
    for name, image in zip(target_names, target_images):
        item = yolov8nface_main_thread.infer_synchronous(image)
        item.name = name
        targets.append(item)

    arcfaceresnet100_main_thread = ArcFaceResnet100(0, "weights/arcfaceresnet100-8.trt", [],
                                                    0.45)

    for target in targets:
        tmp_target = arcfaceresnet100_main_thread.infer_synchronous(target, get_only_embedding=True)
        target.aligned_face_batch = tmp_target.aligned_face_batch
        target.face_embedding_batch = tmp_target.face_embedding_batch

    yolov8nface = YOLOv8nFace(camera_reader_process.output_queue, args.modelpath, conf_threshold=args.confThreshold,
                              iou_threshold=args.nmsThreshold, kill_flag=kill_flag).start()
    arcfaceresnet100 = ArcFaceResnet100(yolov8nface.output_queue, "weights/arcfaceresnet100-8.trt", targets,
                                        0.45, kill_flag=kill_flag).start()

    for item in tqdm.tqdm(arcfaceresnet100):
        item: ArcFaceResnet100Item = item
        stacked_images = []
        for i, name in enumerate(item.matched_names):
            dstimg = cv2.resize(item.aligned_face_batch[i].transpose(1, 2, 0).astype(np.uint8), (300, 300))
            add_text(dstimg, f"{name}-{item.detection_model_time}-{item.model_time}")
            stacked_images.append((dstimg, item.detection_bboxes[i]))
        stacked_images = sorted(stacked_images, key=lambda x: -(x[1][2] * 0.5 + x[1][0] * 0.5))
        stacked_images = [x[0] for x in stacked_images]
        if stacked_images:
            wide_image = np.hstack(stacked_images)  # Stack face_images horizontally
            # yield wide_image
            cv2.imshow('Wide Image', wide_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def add_text(image, text):
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Define font, scale, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2

    # Get the boundary of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Calculate the starting point: centered horizontally and at the top with a small margin
    x = (w - text_width) // 2
    y = text_height + 10  # 10 pixels from the top

    # Add text to the image
    cv2.putText(image, text, (x, y), font, scale, color, thickness)


if __name__ == '__main__':
    main()
