import cv2
import numpy as np
import time
import sys

def transform():

    # Prepare Image and Video
    videoReader = cv2.VideoCapture("./Perspective_Transform/srcfile/test4perspective.mp4")
    img_project = cv2.imread("./Perspective_Transform/srcfile/rl.jpg")
    height, width, _ = img_project.shape

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_id_set = [25, 33, 30, 23]
    aruco_edges = [1, 2, 0, 0]

    while True:
        ret, frame = videoReader.read()

        if ret == False:
            break

        try:
            # Detect the mark corners
            DetectCorners, DetectIds, rejectCandidates = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

            # Adjust order via enumerate to fit the aruco_id_set sequence
            edge_pts = []
            edges = [1, 2, 0, 0]
            for i, marker_id in enumerate(aruco_id_set):
                # First, locate the specific in marker_id index
                index = np.squeeze(np.where(marker_id == DetectIds))[0]
                # print(index, end=" ")
                # Then, add edge point info of each marker into the set
                edge_pts.append(np.squeeze(DetectCorners[index])[edges[i]])

                # print(np.squeeze(DetectCorners[index])])
            # print("")

            distance = np.linalg.norm(edge_pts[0] - edge_pts[1])
            offset = round(distance * 0.02)

            operators = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
            for i, operator in enumerate(operators):
                edge_pts[i][0] += operator[0] * offset
                edge_pts[i][1] += operator[1] * offset

            src_pts = np.array(
                    [[0, 0], [width, 0], [width, height], [0, height]]
                )

            M, mask = cv2.findHomography(
                    src_pts, np.array(edge_pts), cv2.RANSAC
                )

            w, h, _ = frame.shape
            mask_img = cv2.warpPerspective(img_project, M, (h, w))
            ret, mask = cv2.threshold(
                cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV,
            )
            new_frame = cv2.bitwise_and(frame, frame, mask=mask)
            new_frame = cv2.bitwise_or(new_frame, mask_img)

            cv2.imshow("new_frame", new_frame)
            # cv2.waitKey(0)
            #　break

        except Exception as e:
            # exetype, value, traceback = sys.exc_info()
            # print("ExecType: %s, Traceback: %s, Value: %s" % (str(exetype), traceback.tb_lineno, str(value)))
            # 1print('Error opening %s: %s' % (value.filename, value.strerror))
            cv2.imshow("new_frame", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    videoReader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    transform()

# Initialize Aruco Maker
'''( marker_corner,
              marker_ids,
              marker_rejectPts
            )= cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

            edge_pts = []

            for i, mkid in enumerate(aruco_id_set):
                idx = np.squeeze(np.where(marker_ids == mkid))[0]
                edge_pts.append(np.squeeze(marker_corner[idx])[aruco_edges[i]])

            proj_distance = np.linalg.norm(edge_pts[0] - edge_pts[1])
            proj_offset = round(proj_distance * 0.02)

            proj_operators = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
            for i, operator in enumerate(proj_operators):
                edge_pts[i][0] += proj_operators[0] * proj_offset
                edge_pts[i][1] += proj_operators[1] * proj_offset

            src_pts = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]]
            )

            M, mask = cv2.findHomography(
                src_pts, np.array(edge_pts), cv2.RANSAC
            )

            w, h, _ = frame.shape
            mask_img = cv2.warpPerspective(img_project, M, (h, w))

            ret, mask = cv2.threshold(
                cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV,
            )

            new_frame = cv2.bitwise_and(frame, frame, mask=mask)
            new_frame = cv2.bitwise_or(new_frame, mask_img)

            cv2.imshow("Perspective Transform", new_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break'''
