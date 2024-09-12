import cv2
import mediapipe as mp
import pyautogui
import numpy as np

CAMERA_INDEX = 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CLICK_THRESHOLD = 0.01

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame

def detect_face_landmarks(rgb_frame):
    output = face_mesh.process(rgb_frame)
    return output.multi_face_landmarks

def move_mouse(landmarks):
    screen_x = SCREEN_WIDTH * landmarks[1].x
    screen_y = SCREEN_HEIGHT * landmarks[1].y
    pyautogui.moveTo(screen_x, screen_y)

def detect_click(landmarks):
    left_distance = np.linalg.norm(np.array([landmarks[145].x, landmarks[145].y]) - np.array([landmarks[159].x, landmarks[159].y]))
    return left_distance < CLICK_THRESHOLD

cam = cv2.VideoCapture(CAMERA_INDEX)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = process_frame(frame)
    face_landmarks = detect_face_landmarks(rgb_frame)

    if face_landmarks:
        landmarks = face_landmarks[0].landmark
        move_mouse(landmarks)
        if detect_click(landmarks):
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)
