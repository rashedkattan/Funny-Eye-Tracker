import mediapipe as mp
import cv2
import numpy as np
import pygame
import requests
url = "https://thispersondoesnotexist.com/"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
height, width, _ = image.shape
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)
LEFT_EYE_IDS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDS = [33, 160, 158, 133, 153, 144]
def eye_center(landmarks, ids):
    points = [landmarks[i] for i in ids]
    xs = [p.x * width for p in points]
    ys = [p.y * height for p in points]
    return int(np.mean(xs)), int(np.mean(ys))
if not results.multi_face_landmarks:
    print("No face detected.")
    exit()
landmarks = results.multi_face_landmarks[0].landmark
left_center = eye_center(landmarks, LEFT_EYE_IDS)
right_center = eye_center(landmarks, RIGHT_EYE_IDS)
eye_centers = [left_center, right_center]
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Funny Eyes Tracker")
face_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
pupil_radius = 8
max_offset = 10
clock = pygame.time.Clock()
running = True
while running:
    screen.blit(face_surface, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    mx, my = pygame.mouse.get_pos()
    for center in eye_centers:
        dx = mx - center[0]
        dy = my - center[1]
        dist = max(np.hypot(dx, dy), 1e-5)
        offset_x = int(max_offset * dx / dist)
        offset_y = int(max_offset * dy / dist)
        pupil_pos = (center[0] + offset_x, center[1] + offset_y)
        pygame.draw.circle(screen, (255, 255, 255), center, pupil_radius * 2)
        pygame.draw.circle(screen, (0, 0, 0), pupil_pos, pupil_radius)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
#© 2025 Rashed Kattan. All rights reserved.