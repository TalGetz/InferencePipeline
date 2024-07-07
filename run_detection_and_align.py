import cv2
import argparse

import numpy as np
import tqdm

from src.operations.detection.face_detection.align import align_face_np
from src.frame_readers.camera_reader import CameraReader
from src.operations.detection.face_detection.yolov8nface import YOLOv8nFace
from src.runners.tensorrt.trt_runner import TrtRunner


def get_faces(detector, x):
    boxes, scores, classids, kpts = detector.detect(x)
    faces = []
    for kpt, box in zip(kpts, boxes):
        dstimg = align_face_np(x, kpt)
        middle = box[0] + box[2] // 2, box[1] + box[3] // 2
        faces.append((dstimg.transpose(2, 0, 1), middle))
    return faces


def main():
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
    # TrtRunner("weights/")
    embedder = TrtRunner("weights/arcfaceresnet100-8.trt")
    tal = cv2.imread("src/images/tal.png")
    geva = cv2.imread("src/images/geva.png")
    gevald = cv2.imread("src/images/gevald.png")
    targets = {"tal": embedder.infer(get_faces(YOLOv8_face_detector, tal)[0][0])[0],
               "geva": embedder.infer(get_faces(YOLOv8_face_detector, geva)[0][0])[0],
               "gevald": embedder.infer(get_faces(YOLOv8_face_detector, gevald)[0][0])[0]}
    camera = CameraReader()
    frame = camera.get()
    for frame in tqdm.tqdm(camera):
        if frame is None:
            continue
        aligned_faces = get_faces(YOLOv8_face_detector, frame)
        stacked_images = []
        for face, middle in aligned_faces:
            identified = ""
            embeddings = embedder.infer(face)
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
            wide_image = np.hstack(stacked_images)  # Stack images horizontally
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
