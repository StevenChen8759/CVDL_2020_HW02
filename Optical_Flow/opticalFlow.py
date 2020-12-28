import cv2
import numpy as np


def preprocessing():
    # Open Video Reader
    videoReader = cv2.VideoCapture("./Optical_Flow/srcfile/opticalFlow.mp4")

    # Build cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.85
    params.maxCircularity = 0.90
    detector = cv2.SimpleBlobDetector_create(params)

    # Show the video within optical tracker
    print("Show the background Optical Flow - Preprocessing video", end="...")

    kpts = []
    while True:
        ret, frame = videoReader.read()
        if ret == True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            keypoints = detector.detect(frame_gray)

            optic_frame = None
            for p in keypoints:
                x, y = p.pt
                x, y = round(x), round(y)
                kpts.append(p.pt)
                optic_frame = cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
                optic_frame = cv2.line(frame, (x, y-5), (x, y+5), (0, 0, 255), 1)
                optic_frame = cv2.line(frame, (x-5, y), (x+5, y), (0, 0, 255), 1)


            # optic_frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Optical Flow - Preprocessing", optic_frame)
            cv2.waitKey(0)

            break
    print("Done")

    videoReader.release()
    cv2.destroyAllWindows()

    return kpts

def videoTracking(kpts):

    # Open Video Reader
    videoReader = cv2.VideoCapture("./Optical_Flow/srcfile/opticalFlow.mp4")

    # Show the video within optical tracker
    print("Show the background Optical Flow - Video Tracking", end="...")

    # Read First Frame
    ret, frame = videoReader.read()
    frame_prev = frame
    frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

    # Transfer keypoints to numpy ndarray
    kpts_prev = np.asarray(kpts, dtype=np.float32).reshape((len(kpts), 1, 2))

    # Create a mask to record the path
    mask = np.zeros_like(frame_prev)

    # Edit cv2.calcOpticalFlowPyrLK - lk_kwargs
    lk_kwargs = dict(winSize = (21, 21), maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        # Read Current Frame and Convert to gray scale
        ret, frame_curr = videoReader.read()
        if ret == True:
            frame_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Do cv2.calcOpticalFlowPyrLK
        kpts_curr, stat, err = cv2.calcOpticalFlowPyrLK(frame_prev_gray, frame_curr_gray,
                                                        kpts_prev, None, **lk_kwargs)

        # Select well-tracked point in kpts_curr
        # wt_kpts_prev = kpts_prev[stat == 1]
        # wt_kpts_curr = kpts_curr[stat == 1]

        for (curr, prev) in zip(kpts_curr, kpts_prev):
            a, b = curr.ravel()
            c, d = prev.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 0, 200), 3)
            a = round(a)
            b = round(b)
            frame_curr = cv2.rectangle(frame_curr, (a - 5, b - 5), (a + 5, b + 5), (0, 0, 255), 1)
            frame_curr = cv2.line(frame_curr, (a, b - 5), (a, b + 5), (0, 0, 255), 1)
            frame_curr = cv2.line(frame_curr, (a - 5, b), (a + 5, b), (0, 0, 255), 1)

        frame_show = cv2.add(frame_curr, mask)
        cv2.imshow("Optical Flow - Video Tracking", frame_show)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

        # Current frame/kpts assign to Previous frame/kpts
        frame_prev_gray = frame_curr_gray.copy()
        kpts_prev = kpts_curr

    print("Done")

    videoReader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    kpts = preprocessing()
    videoTracking(kpts)