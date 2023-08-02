import cv2


class FaceDetection:
    def __init__(self) -> None:
        """Initialize the FaceDetection class"""
        self.face_cascade = cv2.CascadeClassifier("facedetection.xml")
        self.eye_cascade = cv2.CascadeClassifier("eye.xml")

    def _video_capture_loop(self) -> None:
        """Capture video and perform face and eye detection."""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=3)
            eyes = self.eye_cascade.detectMultiScale(gray)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 5)
                for x1, y1, w, h in eyes:
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (120, 0, 255), 5)
            cv2.imshow("Face and Eye Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detection(self) -> None:
        """Run the face and eye detection algorithm."""
        self._video_capture_loop()


if __name__ == "__main__":
    detect = FaceDetection()
    detect.detection()
