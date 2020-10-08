"""
Face and Eye detection using OpenCV
"""
import cv2

class FaceDetection:
	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier('facedetection.xml')
		self.eye_cascade = cv2.CascadeClassifier('eye.xml')
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.cap = cv2.VideoCapture(0)


	def detection(self):
		while (self.cap.isOpened()):
			# Capture the frame
			ret,frame = self.cap.read()
			# Convert the image to grayscale
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			# Get the coordinates of the face
			faces = self.face_cascade.detectMultiScale(gray,1.4,3)
			# Get the eye coordinates
			eyes = self.eye_cascade.detectMultiScale(gray)
			# Iterate through the coordinates
			for (x,y,w,h) in faces:
				# Draw the rectangle on the frame
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),5)
				# Put the text around the frame
				#cv2.putText(frame,"Obj",(x-10,y-10),self.font,1,(255,0,0),5,cv2.LINE_AA)
				# Iterate through the eye coordinate
				for x1,y1,w,h in eyes:
					cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,255),5)
			cv2.imshow("Result",frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		self.cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	detect = FaceDetection()
	detect.detection()
	