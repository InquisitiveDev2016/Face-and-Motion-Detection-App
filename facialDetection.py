from cv2 import cv2

#Creating a Cascade Classifer Object
face_cascade = cv2.CascadeClassifier("/Users/alihaider/Documents/Python/frontalFace10/haarcascade_frontalface_default.xml")

#Reading the img
img = cv2.imread("/Users/alihaider/Documents/Python/photo.jpg", 1)

#Reading the img as a gray scale img
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Search the coordinates of the img
faces = face_cascade.detectMultiScale(gray_img, scaleFactor= 1.05, minNeighbors=5)

print(type(faces))
print(faces)

#Creating the rectangular face box
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0), 3)


cv2.imshow("Gray", img)

cv2.waitKey(0)
