#!/usr/bin/env python
from datetime import datetime
from sys import argv
import cv2
from progress import update_progress

if len(argv) == 1:
    print(f"Usage: {argv[0]} <video file path>")
    exit(1)


VIDEO_PATH = argv[1]


from functions import *

rotate = False
skip_frames = 10 # i.e. at 30 fps, this advances one second

video_capture = cv2.VideoCapture(VIDEO_PATH)
# print(video_capture.set(cv2.CAP_PROP_FPS, 5))

video_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
video_width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps    = video_capture.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {video_frames}, height: {video_height}, width: {video_width}, "
	f"FPS: {video_fps}, Skipping frames: {skip_frames}")

# Initialize some variables
face_locations = []
face_encodings = []
count = 0


while True:
	current_frame_count = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
	# print(f"Processing frame {current_frame_count}...")

	update_progress(current_frame_count*100/video_frames, f"Processing video {current_frame_count}")

	# Grab a single frame of video
	success, frame = video_capture.read()

	if rotate:
		frame = cv2.rotate(frame, cv2.ROTATE_180)

	# if frame empty exit or current frame is equal to last frame
	if not success or current_frame_count >= video_frames:
		break

	times = 0.9
	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=times, fy=times)
	# small_frame = frame.copy()

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]
	
	# Find all the faces and face encodings in the current frame of video
	face_locations = get_face_locations(rgb_small_frame)
	face_encodings = get_face_encodings(rgb_small_frame, face_locations)

	for face_encoding in face_encodings:
		# Or instead, use the known face with the smallest distance to the new face
		face_distances = face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		# print(face_distances[best_match_index], known_face_labels[best_match_index])
		if face_distances[best_match_index] <= threshold:
			name = known_face_labels[best_match_index]
			export_data.append((name, datetime.now()))

	if skip_frames:
		count += skip_frames
		video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)

save_export()
