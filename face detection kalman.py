import  cv2
import numpy as np



video_capture = cv2.VideoCapture('face_video.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def faceDetection(video):
    ret, frame = video.read()
    if ret == False:
        return

    # read a frame each time and detect face in that frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # findin eyes and mouth in faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
    while (1):
        #read the next frames , return if neccessary
        ret, frame = video.read()
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        #if no face detected set the c r w h zero
        if len(faces) == 0:
            c, r, w, h = (0, 0, 0, 0)
        else:
            c, r, w, h = faces[0]

         #if face exists
        if c != 0 and w != 0 and r != 0 and h != 0:
        #draw rectange in output area (face position)
         cv2.rectangle(frame, ((int)(x-h/2), (int)(y-w/2)), ((int)(x+h/2), (int)(y+w/2)), (0, 255, 255), 2)
        print("face tracker without filter press q to quit and see the output with kalman filter!")
        cv2.imshow('face-tracker without kalman filter', frame)
          #exit the face tracker by entering 'q' on keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def Kalmantracker(video):
    frameCounter = 0
    #read frame from video file
    ret, frame = video.read()
    if ret == False:
        return

    #read a frame each time and detect face in that frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    #face exists?
    if len(faces) == 0:
        c, r, w, h = (0, 0, 0, 0)
    else:
        c, r, w, h = faces[0]
    #after finding face add 1 to frame counter
    frameCounter = frameCounter + 1
    #state is initial position now
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
    kalman = cv2.KalmanFilter(4, 2, 0)

#speed model
    kalman.transitionMatrix = np.array([[1., 0., .2, 0.],[0., 1., 0., .2],[0., 0., 1., 0.],[0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    #init tracker
    # now read the remaining frames
    while (1):
        #read the next frames , return if neccessary
        ret, frame = video.read()
        if ret == False:
            break
        #after reading the frame  , predict !
        prediction = kalman.predict()
        #find faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        #if no face detected set the c r w h zero
        if len(faces) == 0:
            c, r, w, h = (0, 0, 0, 0)
        else:
            c, r, w, h = faces[0]

        #measuread parameteres
        measurement = np.array([c + w / 2, r + h / 2], dtype='float64')
        #output is what we predixted temporary
        output = prediction
         #if face exists
        if c != 0 and w != 0 and r != 0 and h != 0:
            #correct the measured parameters and put it in output
            posterior = kalman.correct(measurement)
            output = posterior
        (x,y,z,q) = output
        #draw rectange in output area (face position)
        cv2.rectangle(frame, ((int)(x-h/2), (int)(y-w/2)), ((int)(x+h/2), (int)(y+w/2)), (0, 255, 255), 2)
        frameCounter = frameCounter + 1

        cv2.imshow('face-tracker', frame)
        print("face tracker with kalman filter")
          #exit the face tracker by entering 'q' on keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

faceDetection(video_capture)
Kalmantracker(video_capture)
cv2.destroyAllWindows()
