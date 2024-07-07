import cv2
import argparse

import numpy as np
import tqdm

from src.operations.detection.face_detection.align import align_face_np
from src.frame_readers.camera_reader import CameraReader
from src.operations.detection.face_detection.yolov8nface import YOLOv8nFace

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='face.jpeg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.onnx',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize the YOLOv8_face detector
    YOLOv8_face_detector = YOLOv8nFace(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)

    camera = CameraReader()
    frame = camera.get()
    for frame in tqdm.tqdm(camera):
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(frame)

        # frame = YOLOv8_face_detector.draw_detections(frame, boxes, scores, kpts)

        stacked_images = []
        for kpt in kpts:
            dstimg = align_face_np(frame, kpt)
            dstimg = cv2.resize(dstimg, (400, 400))
            stacked_images.append(dstimg)

        if stacked_images:
            wide_image = np.hstack(stacked_images)  # Stack images horizontally
            cv2.imshow('Wide Image', wide_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
