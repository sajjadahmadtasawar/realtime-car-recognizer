import cv2


car_trained_data = cv2.CascadeClassifier('cars.xml')


video = cv2.VideoCapture('cars.mp4')

while True:
    (success,frame) = video.read()

    if success:
        grayScaled_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        car_coordinates = car_trained_data.detectMultiScale(grayScaled_video)
        for (x,y,w,h) in car_coordinates:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,266,0),2)
    else:
        break
    cv2.imshow('car detector',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
video.release()       



