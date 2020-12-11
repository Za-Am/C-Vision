import cv2 

cascade_face = 'cascades/haarcascade_frontalface_default.xml'
cascade_eye = 'cascades/haarcascade_eye.xml'

face_classifier = cv2.CascadeClassifier(cascade_face)
eye_classifier = cv2.CascadeClassifier(cascade_eye)

image = cv2.imread('images/svalley.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)
print (faces)

if faces.any():
	for (x, y, w, h) in faces:
		print (len(faces))
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0), 3)
		cv2.imshow("Face Detection", image)
		cv2.waitKey(0)

		area_gray = gray[y:y+h, x:x+w]
		area_original = image[y:y+h, x:x+w]

		eyes = eye_classifier.detectMultiScale(area_gray)
		print (eyes)
		print (len(eyes))
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(area_original, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
			cv2.imshow("Face Detection", image)
			cv2.waitKey(0)



elif faces is ():
	print ("No Faces Found")		

cv2.destroyAllWindows()