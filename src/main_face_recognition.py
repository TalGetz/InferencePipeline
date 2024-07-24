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
from src.models.recognition.face_recognition.item import FaceRecognitionItem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_model_path', type=str, default='weights/yolov8n-face.trt')
    parser.add_argument('--recognition_model_path', type=str, default='weights/arcfaceresnet100-8.trt')
    parser.add_argument('--detection_confidence_threshold', default=0.65, type=float)
    parser.add_argument('--detection_nms_threshold', default=0.5, type=float)
    parser.add_argument('--face_recognition_threshold', default=0.5, type=float)
    parser.add_argument("--targets_folder_path", default="data/target_face_images/", type=str)
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except:
        print("Not using start method spawn, probably on windows.")

    kill_flag = threading.Event()
    if config.DEBUG:
        run(kill_flag, args.targets_folder_path, args.detection_model_path, args.recognition_model_path,
            args.detection_confidence_threshold, args.detection_nms_threshold, args.face_recognition_threshold)
    else:
        try:
            run(kill_flag, args.targets_folder_path, args.detection_model_path, args.recognition_model_path,
                args.detection_confidence_threshold, args.detection_nms_threshold, args.face_recognition_threshold)
        except:
            pass

    kill_flag.set()
    print("shutting down: main")
    time.sleep(3)


def run(kill_flag, targets_folder_path, detection_model_path, recognition_model_path,
        detection_confidence_threshold, detection_nms_threshold, face_recognition_threshold):
    for wide_image in wide_image_generator(kill_flag, targets_folder_path, detection_model_path, recognition_model_path,
                                           detection_confidence_threshold, detection_nms_threshold,
                                           face_recognition_threshold):
        cv2.imshow('Wide Image', wide_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def wide_image_generator(kill_flag, targets_folder_path, detection_model_path, recognition_model_path,
                         detection_confidence_threshold, detection_nms_threshold, face_recognition_threshold):
    camera_reader_process = CameraReaderProcess(kill_flag=kill_flag).start()
    targets = gather_targets(targets_folder_path, detection_model_path, recognition_model_path,
                             detection_confidence_threshold, detection_nms_threshold)

    yolov8nface = YOLOv8nFace(camera_reader_process.output_queue, model_path=detection_model_path,
                              conf_threshold=detection_confidence_threshold,
                              iou_threshold=detection_nms_threshold, kill_flag=kill_flag).start()
    arcfaceresnet100 = ArcFaceResnet100(yolov8nface.output_queue, recognition_model_path, targets,
                                        face_recognition_threshold, kill_flag=kill_flag).start()

    for item in tqdm.tqdm(arcfaceresnet100):
        wide_image = create_merged_aligned_image(item)
        if wide_image is not None:
            yield wide_image


def gather_targets(targets_folder_path, detection_model_path, recognition_model_path,
                   target_detection_confidence_threshold, target_detection_nms_threshold):
    yolov8nface_main_thread = YOLOv8nFace(0, detection_model_path,
                                          conf_threshold=target_detection_confidence_threshold,
                                          iou_threshold=target_detection_nms_threshold)
    target_file_names = os.listdir(targets_folder_path)
    target_names = [x[:-4] for x in target_file_names if x.endswith(".jpg") or x.endswith(".png")]
    target_images = [cv2.imread(str(Path(targets_folder_path) / file_name)) for file_name in target_file_names]
    targets = []
    for image in target_images:
        item = yolov8nface_main_thread.infer_synchronous(image)
        targets.append(item)

    arcfaceresnet100_main_thread = ArcFaceResnet100(0, recognition_model_path, [],
                                                    0.45)

    for name, target in zip(target_names, targets):
        tmp_target = arcfaceresnet100_main_thread.infer_synchronous(target)
        target.aligned_face_batch = tmp_target.aligned_face_batch
        target.face_embedding_batch = tmp_target.face_embedding_batch
        target.name = name

    return targets


def create_merged_aligned_image(item: FaceRecognitionItem):
    stacked_images = []
    for i, name in enumerate(item.matched_names):
        dstimg = cv2.resize(item.aligned_face_batch[i].transpose(1, 2, 0).astype(np.uint8), (300, 300))
        add_text(dstimg, name)
        stacked_images.append((dstimg, item.detection_bboxes[i]))
    stacked_images = sorted(stacked_images, key=lambda x: -(x[1][2] * 0.5 + x[1][0] * 0.5))
    stacked_images = [x[0] for x in stacked_images]
    if stacked_images:
        # Stack face_images vertically
        return np.hstack(stacked_images)
    return None


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
