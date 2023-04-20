import cv2
import numpy as np

# Set the known width of the QR code in inches
KNOWN_WIDTH = 2.0

# Load the webcam
cap = cv2.VideoCapture(0)

# Initialize the camera matrix
camera_matrix = np.array([[6.76512085e+03, 0.00000000e+00, 9.93144974e+02],
                          [0.00000000e+00, 6.77437549e+03, 1.14477852e+03],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Initialize the distortion coefficients
dist_coeffs = np.array([-6.96445690e-02, -4.23134320e-01, -1.62658558e-03,
                        1.66192427e-03, 2.43709853e+00])

# Define the aruco dictionary
#aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Define the aruco parameters
aruco_params = cv2.aruco.DetectorParameters_create()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # If markers are detected, draw the bounding boxes and calculate the distance to the camera
    if len(corners) > 0:
        # Estimate the pose of the markers
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, KNOWN_WIDTH, camera_matrix, dist_coeffs)

        # Draw the bounding boxes
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw the axes of the markers
        for i in range(len(ids)):
            #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], KNOWN_WIDTH / 2)
             corners_i = corners[i][0]
             axis_points = np.float32([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]).reshape(-1, 3)
             img_points, _ = cv2.projectPoints(axis_points, rvec[i], tvec[i], camera_matrix, dist_coeffs)
             img_points = np.int32(img_points).reshape(-1, 2)
             cv2.drawContours(frame, [img_points[:4]], -1, (0, 0, 255), 2) # x-axis (red)
             cv2.drawContours(frame, [img_points[[0, 2]]], -1, (0, 255, 0), 2) # y-axis (green)
             cv2.drawContours(frame, [img_points[[0, 3]]], -1, (255, 0, 0), 2) # z-axis (blue)
        # Calculate the distance to the markers
        distances = []
        for i in range(len(ids)):
            distance = KNOWN_WIDTH * camera_matrix[0, 0] / (2 * corners[i][0][0][0] * np.tan(31 * np.pi / 180 / 2))
            distances.append(distance)
            cv2.putText(frame, f"Distance: {distance:.2f} inches", (20, 40+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
