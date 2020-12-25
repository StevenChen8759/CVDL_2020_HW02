import cv2
import numpy as np

def bgsub():
    # Open Video Reader
    videoReader = cv2.VideoCapture("./Background_Subtraction/srcfile/bgsub.mp4")

    # Fetch top 50 frame of the video
    print("Fetch top 50 frames", end='...')
    cnt = 0
    frameCollection = []
    while cnt != 50:
        ret, frame = videoReader.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frameCollection.append(frame_gray)
            cnt = cnt + 1

    print("Done")

    # Generate Gaussian Model based on the image read upward
    print("Generate Gaussian Model", end='...')
    # TODO: Build Gaussian Model
    print("Done")


    print("Show the background subtraction video", end="...")
    videoReader.set(1, 0)  # Reset the reading frame to the head
    while True:
            ret, frame = videoReader.read()
            if ret == True:
                # TODO: Put Gaussian processed frame into 2nd input
                newframe = np.concatenate((frame,frame), axis=1)
                cv2.imshow("Background Subtraction", newframe)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    print("Done")

    videoReader.release()
    cv2.destroyAllWindows()
    '''while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)'''


if __name__ == "__main__":
    bgsub()