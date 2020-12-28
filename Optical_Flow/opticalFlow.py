import cv2
import numpy as np

def preprocessing():
    # Open Video Reader
    videoReader = cv2.VideoCapture("./Optical_Flow/srcfile/opticalFlow.mp4")

    # Build cv2.SimpleBlobDetector
    # TODO: Adjust parameters to get the optimal detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity=True
    params.minCircularity = 0.6
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 110
    detector = cv2.SimpleBlobDetector_create(params)

    # Show the video within optical tracker
    print("Show the background Optical Flow - Preprocessing video", end="...")
    while True:
            ret, frame = videoReader.read()
            if ret == True:
                keypoints = detector.detect(frame)
                optic_frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("Optical Flow - Preprocessing", optic_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    print("Done")

    videoReader.release()
    cv2.destroyAllWindows()

def videoTracking():
    pass

if __name__ == "__main__":
    preprocessing()