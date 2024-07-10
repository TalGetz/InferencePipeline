import argparse
import time

import cv2
import numpy as np
import tqdm

from src.frame_readers.camera_reader_process import CameraReaderProcess
from src.models.detection.face_detection.yolov8nface.item import YOLOv8nFaceItem
from src.models.detection.face_detection.yolov8nface.yolov8nface import YOLOv8nFace
from src.models.recognition.face_recognition.arcfaceresnet100.arcfaceresnet100 import ArcFaceResnet100
from src.models.recognition.face_recognition.arcfaceresnet100.target import ArcFaceResnet100Target
from src.utils.align import align_face_np


def get_faces(item: YOLOv8nFaceItem):
    boxes, scores, classids, kpts = item.det_bboxes, item.det_conf, item.det_classid, item.landmarks
    faces = []
    for kpt, box in zip(kpts, boxes):
        dstimg = align_face_np(item.frame, kpt)
        middle = box[0] + box[2] // 2, box[1] + box[3] // 2
        faces.append((dstimg.transpose(2, 0, 1), middle))
    return faces


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='face.jpeg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.trt',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize the YOLOv8_face detector

    camera_reader_process = CameraReaderProcess().start()
    yolov8nface_main_thread = YOLOv8nFace(0, 1, args.modelpath, conf_threshold=args.confThreshold,
                                          iou_threshold=args.nmsThreshold)

    target_names = ["tal", "geva"]
    target_images = [cv2.imread(f"data/face_images/{name}.png") for name in target_names]
    targets = [
        ArcFaceResnet100Target(
            image,
            yolov8nface_main_thread.infer_synchronous(image).landmarks,
            name=name
        )
        for name, image in zip(target_names, target_images)
    ]
    arcfaceresnet100_main_thread = ArcFaceResnet100(0, 1,
                                                    "weights/arcfaceresnet100-8.trt", targets, 0.3)
    for target in targets:
        target.face_embedding_batch = arcfaceresnet100_main_thread.infer_synchronous(target,
                                                                                     get_only_embedding=True).face_embedding_batch

    yolov8nface = YOLOv8nFace(camera_reader_process.output_queue, 1, args.modelpath, conf_threshold=args.confThreshold,
                              iou_threshold=args.nmsThreshold).start()
    arcfaceresnet100 = ArcFaceResnet100(yolov8nface.output_queue, 1,
                                        "weights/arcfaceresnet100-8.trt", targets, 0.3).start()

    time.sleep(1)
    for item in tqdm.tqdm(arcfaceresnet100):
        print(item)
        continue

        aligned_faces = get_faces(item)
        stacked_images = []
        for face, middle in aligned_faces:
            identified = ""
            embeddings = embedder.infer([face])
            for embedding in embeddings:
                for target in targets:
                    similarity = (embedding / np.linalg.norm(embedding)) @ (
                            targets[target] / np.linalg.norm(targets[target])).T
                    if similarity > 0.3:
                        identified += target + " "

            dstimg = cv2.resize(face.transpose(1, 2, 0).astype(np.uint8), (400, 400))
            if identified:
                add_text(dstimg, identified)
            stacked_images.append((dstimg, middle))
        stacked_images = sorted(stacked_images, key=lambda x: -x[1][0])
        stacked_images = [x[0] for x in stacked_images]
        if stacked_images:
            wide_image = np.hstack(stacked_images)  # Stack face_images horizontally
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
