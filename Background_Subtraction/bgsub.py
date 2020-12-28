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
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameCollection.append(frame_gray)
            cnt = cnt + 1

    print("Done")

    # Generate Gaussian Model based on the image read upward
    print("Generate Gaussian Model", end='...')

    # Read frame as numpy array
    frame_np_input = np.array([x for x in frameCollection])

    # Fetch width and height information, then create mean and std matrix
    pixel_set = np.moveaxis(frame_np_input, 0, -1)
    height, width, n = pixel_set.shape
    mat_mean = np.zeros((height, width))
    mat_std = np.zeros((height, width))

    # Traverse each pixels, then calculate mean and std for each pixel
    for i in range(height):
        for j in range(width):
            pixels = pixel_set[i][j]
            mean = np.mean(pixels)
            std = np.std(pixels)
            std = std if std > 5 else 5
            mat_mean[i][j] = mean
            mat_std[i][j] = std

    print("Done")


    print("Show the background subtraction video", end="...")
    videoReader.set(1, 0)  # Reset the reading frame to the head
    while True:
            ret, frame = videoReader.read()
            if ret == True:
                # Filter by Gaussian Filter, then output as grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gaussian = frame_gray - mat_mean
                frame_gaussian = (frame_gaussian > 5 * mat_std).astype(np.uint8) * 255

                frame_show = cv2.hconcat([frame, np.stack((frame_gaussian,) * 3, axis=-1)])
                cv2.imshow("Background Subtraction", frame_show)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    print("Done")

    videoReader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    bgsub()