import argparse
import multiprocessing
import threading
import time

import cv2
import tqdm

from src import config
from src.frame_readers.camera_reader_process import CameraReaderProcess
from src.models.detection.item import DetectionItem
from src.models.detection.yolo.yolov10.coco_label_names import COCO_LABEL_NAMES_DICT
from src.models.detection.yolo.yolov10.yolov10 import YOLOv10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_model_path', type=str, default='weights/yolov10x.trt')
    parser.add_argument('--detection_confidence_threshold', default=0.65, type=float)
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except:
        print("Not using start method spawn, probably on windows.")

    kill_flag = threading.Event()
    if config.DEBUG:
        run(kill_flag, args.detection_model_path, args.detection_confidence_threshold)
    else:
        try:
            run(kill_flag, args.detection_model_path, args.detection_confidence_threshold)
        except:
            pass

    kill_flag.set()
    print("shutting down: main")
    time.sleep(3)


def run(kill_flag, detection_model_path, detection_confidence_threshold):
    for wide_image in wide_image_generator(kill_flag, detection_model_path, detection_confidence_threshold):
        cv2.imshow('Wide Image', wide_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def wide_image_generator(kill_flag, detection_model_path, detection_confidence_threshold):
    camera_reader_process = CameraReaderProcess(kill_flag=kill_flag).start()

    yolov10 = YOLOv10(camera_reader_process.output_queue, model_path=detection_model_path,
                      conf_threshold=detection_confidence_threshold, kill_flag=kill_flag).start()

    for item in tqdm.tqdm(yolov10):
        image = create_bbox_image(item)
        if image is not None:
            yield image


def create_bbox_image(item: DetectionItem):
    frame = item.frame
    for i, class_ids in enumerate(item.detection_class_id):
        class_name = COCO_LABEL_NAMES_DICT[class_ids + 1]
        x1, y1, x2, y2 = item.detection_bboxes[i]
        add_text(frame, class_name, (x2+x1)//2, y1)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def add_text(image, text, x, y):
    # Get image dimensions

    # Define font, scale, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2

    # Get the boundary of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Calculate the starting point: centered horizontally and at the top with a small margin
    x = x - text_width // 2
    y = y + text_height + 10  # 10 pixels from the top

    # Add text to the image
    cv2.putText(image, text, (x, y), font, scale, color, thickness)


if __name__ == '__main__':
    main()
