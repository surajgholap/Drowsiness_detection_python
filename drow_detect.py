import cv
import time
import sys
import os


def DetectFace(image, faceCascade, eyeCascade):
    min_size = (20, 20)
    image_scale = 2
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    # Allocate the temporary images
    gray_scale = cv.CreateImage((image.width, image.height), 8, 1)
    small_Image = cv.CreateImage(
            (
                cv.Round(image.width / image_scale),
                cv.Round(image.height / image_scale)
            ), 8, 1)
    # Convert color input image to grayscale
    cv.CvtColor(image, gray_scale, cv.CV_BGR2GRAY)
    # Scale input image for faster processing
    cv.Resize(gray_scale, small_Image, cv.CV_INTER_LINEAR)
    # Equalize the histogram
    cv.EqualizeHist(small_Image, small_Image)
    faces = cv.HaarDetectObjects(                        # Haarcascade objects for face detection
            small_Image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

    eyes = cv.HaarDetectObjects(small_Image, eyeCascade,  # Haarcascade objects for eyes detection
                                 cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors,
                                 haar_flags, min_size)
    # If face is detected
    if faces:
        for ((x, y, w, h), n) in faces:
            # the input to cv.HaarDetectObjects was resized, so scale the
            # bounding box of each face and convert it to two CvPoints
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)
        # if eyes detected once face is detected    
        if eyes:
            for ((x, y, w, h), n) in eyes:
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each eye and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(image, pt1, pt2, cv.RGB(0, 255, 0), 5, 8, 0)
        else:
            os.system("scrot ab.png")
            os.system("python detect.py")
        #   os.system("mplayer 1.mp3")
    else:
        os.system("scrot noface.png")
        print('User not detected')
    #  os.system("mplayer 1.mp3")
    return image

# Capture from webcam
capture = cv.CaptureFromCAM(0)
# Capture from any video file
# capture = cv.CaptureFromFile("test.avi")
# To load Frontalface and eyes xml
face_Cascade = cv.Load("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
eye_Cascade = cv.Load("/usr/share/opencv/haarcascades/haarcascade_eye.xml")
while (cv.WaitKey(15) == -1):
    img = cv.QueryFrame(capture)
    image = DetectFace(img, face_Cascade, eye_Cascade)
    cv.ShowImage("face detection test", image)
cv.ReleaseCapture(capture)
