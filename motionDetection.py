from cv2 import cv2
import time


video = cv2.VideoCapture(0)

#The first  and second frame
ret, frame1 = video.read()
ret, frame2 = video.read()

while video.isOpened():

    #finding out the abs diff between the first and second frame
    diff = cv2.absdiff(frame1, frame2)

    #Converting the diff into a gray scale mode
    #It is easier to find the contours of a grey scale img than a full colour
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #blurring the grey scale frame
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    #Evaluating the threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)

    #Finding out the contour
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)

        #If the area is < 700, we do not put a contour around it, so we are trying to avoid contouring anything thats not a person
        if cv2.contourArea(contour) < 700:
            continue

        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 3)

    #-1 applies all the contours
    cv2.drawContours(frame1, contours, -1, (0,255, 0), 2)


    cv2.imshow("feed", frame1)

    frame1 = frame2
    
    #reading the new frame into the 2nd variable
    ret, frame2 = video.read()

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
