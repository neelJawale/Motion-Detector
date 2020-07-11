##Implementation of Webcam Motion Detector using Python

import cv2, time, pandas
from datetime import datetime

#Initializing firstFrame variable
firstFrame = None
#List for moving objects
statusList = [None,None]
times = []
#Initializing data frame
df = pandas.DataFrame(columns=["Start","End"])

#Capturing video through webcam
video = cv2.VideoCapture(0)

#Infinite while loop to treat stack of image frames as a video
while True:
    check, frame = video.read()
    #Status of motion is 0
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Implementation of Gaussian Blur
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    deltaFrame = cv2.absdiff(firstFrame, gray)

    """
    If the difference in intensity between static background
    and the current frame is less than threshold (here 35),
    it will show whit colour.
    """
    threshFrame = cv2.threshold(deltaFrame, 35, 255, cv2.THRESH_BINARY)[1]
    threshFrame = cv2.dilate(threshFrame, None, iterations = 3)

    #Find contour of moving object(s)
    contours,heirarchy = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
         #Make a green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    statusList.append(status)

    statusList=statusList[-2:]

    #Append start time of motion
    if statusList[-1] == 1 and statusList[-2] == 0:
        times.append(datetime.now())
    #Append end time of motion
    if statusList[-1] == 0 and statusList[-2] == 1:
        times.append(datetime.now())

    """
    Display all four frames viz. Gray Frame, Delta Frame
    Thresh Frame and Color frame
    """

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", deltaFrame)
    cv2.imshow("Threshold Frame", threshFrame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    #Enter 's' to terminate the whole process
    if key == ord('s'):
        if status == 1:
            times.append(datetime.now())
        break

print(statusList)
print(times)

#Append the time of motion
for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

#Store the movements' log in a CSV file
df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows
