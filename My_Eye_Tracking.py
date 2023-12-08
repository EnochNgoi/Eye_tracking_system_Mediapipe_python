# Import Libraries
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import pyttsx3

# Utility Functions
# Function of blinking
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2) # Dinstance of vector 1 and 2 

# Function of head down
def calculate_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)) # Find the similarities of both graph 
    angle = np.degrees(np.arccos(cosine_angle)) # Covert the similaritis into degree 
    return angle

map_face_mesh = mp.solutions.face_mesh

# Define Facial Landmark Indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

TOP_EYE_IDX = 159
BOTTOM_EYE_IDX = 145

# Initialize Variables
blink_count = 0
blink_count_minute = 0
last_reset_time = time.time()
BLINK_RATE_THRESHOLD = 25
distance_threshold = None
DISTANCE_THRESHOLD_FACTOR = 0.6
frame_counter = 0
AVERAGE_FRAMES = 5
average_distances = []

ALERT_THRESHOLD_SECONDS = 3

head_down_angle_threshold = 78 # Fine tune to get this value 
HEAD_DOWN_ALERT_SECONDS = 5  # Change to 5 seconds
head_down_frame_counter = 0

# Open Video Capture Device
cap = cv.VideoCapture(0)

start_time = time.time()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

with map_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    # Enter Main Loop
    while True:
        elapsed_time = time.time() - start_time

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        result = face_mesh.process(rgb_frame) # Facial Feature Detection

        if result.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in result.multi_face_landmarks[0].landmark] # Get Facial Landmarks Coordinates
            )

            cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)

            vertical_distance = calculate_distance(mesh_points[TOP_EYE_IDX], mesh_points[BOTTOM_EYE_IDX])

            # Running Average Calculation
            if len(average_distances) < AVERAGE_FRAMES:
                average_distances.append(vertical_distance)
            else:
                average_distances.pop(0)
                average_distances.append(vertical_distance)

                # Setting the Distance Threshold
                if distance_threshold is None:
                    average_distance = np.mean(average_distances)
                    distance_threshold = average_distance * DISTANCE_THRESHOLD_FACTOR

                # Blink Detection
                if vertical_distance < distance_threshold:
                    frame_counter += 1
                else:
                    if frame_counter > 0:
                        blink_count += 1
                        blink_count_minute += 1
                    frame_counter = 0

                cv.putText(frame, "Blinks: {}".format(blink_count), (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Alert if eyes are closed for more than ALERT_THRESHOLD_SECONDS
                if frame_counter > ALERT_THRESHOLD_SECONDS * 30:
                    engine.say("Alert: Your eyes have been closed for more than 3 seconds!")
                    engine.runAndWait()
                    frame_counter = 0

                # Reset blink count every minute and check for blink rate threshold
                if time.time() - last_reset_time >= 60:
                    if blink_count_minute > BLINK_RATE_THRESHOLD:
                        engine.say("Alert: Your blink rate has exceeded 25 times per minute!")
                        engine.runAndWait()

                    blink_count_minute = 0
                    last_reset_time = time.time()

                        # Head movement detection
            left_eye_angle = calculate_angle(mesh_points[LEFT_EYE[0]], mesh_points[TOP_EYE_IDX], mesh_points[BOTTOM_EYE_IDX])
            right_eye_angle = calculate_angle(mesh_points[RIGHT_EYE[0]], mesh_points[TOP_EYE_IDX], mesh_points[BOTTOM_EYE_IDX])
            avg_eye_angle = (left_eye_angle + right_eye_angle) / 2

            if avg_eye_angle > head_down_angle_threshold:
                head_down_frame_counter += 1
            else:
                head_down_frame_counter = 0

            # Alert if head is down for more than HEAD_DOWN_ALERT_SECONDS
            if head_down_frame_counter > HEAD_DOWN_ALERT_SECONDS * 30:
                engine.say("Alert: Your head is down for more than {} seconds!".format(HEAD_DOWN_ALERT_SECONDS))
                engine.runAndWait()
                head_down_frame_counter = 0

        cv.putText(frame, "Elapsed Time: {:.2f} seconds".format(elapsed_time), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the average eye angle on the frame
        cv.putText(frame, "Average Eye Angle: {:.2f} degrees".format(avg_eye_angle), (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

