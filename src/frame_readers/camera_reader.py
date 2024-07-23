import cv2


class CameraReader:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Camera with index {camera_index} cannot be opened.")

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    def get(self):
        ret, frame = self.cap.read()
        while not ret:
            ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()


# Example usage
if __name__ == "__main__":
    camera = CameraReader()
    while True:
        frame = camera.get()
        cv2.imshow("Video Frame", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
