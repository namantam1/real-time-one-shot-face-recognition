#!/usr/bin/env python
from sys import argv
import cv2

if len(argv) == 1:
    print(f"Usage: {argv[0]} <video file path>")
    exit(1)


# VIDEO_PATH = "C:\\Users\\Naman Tamrakar\\Videos\\28.01.2022_15.36.07_REC.mp4"
VIDEO_PATH = argv[1]

from functions import *

rotate = False
skip_frames = 20 # i.e. at 30 fps, this advances one second

video_capture = cv2.VideoCapture(VIDEO_PATH)


video_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
video_width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps    = video_capture.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {video_frames}, height: {video_height}, width: {video_width}, "
	f"FPS: {video_fps}, Skipping frames: {skip_frames}")


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
count = 0

while True:
	current_frame_count = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
	print(f"Processing frame {current_frame_count}...")

	# Grab a single frame of video
	success, frame = video_capture.read()

	if rotate:
		frame = cv2.rotate(frame, cv2.ROTATE_180)

	# if frame empty exit
	if not success or current_frame_count >= video_frames:
		break

	times = 0.8
	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=times, fy=times)
	# small_frame = frame.copy()

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]
	# print(rgb_small_frame.shape)
	
	# Find all the faces and face encodings in the current frame of video
	face_locations = get_face_locations(rgb_small_frame)
	face_encodings = get_face_encodings(rgb_small_frame, face_locations)
	# print(len(face_encodings))
	face_names = []
	for face_encoding in face_encodings:
		# Or instead, use the known face with the smallest distance to the new face
		face_distances = face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		# if matches[best_match_index]:
		if face_distances[best_match_index] <= threshold:
			name = known_face_labels[best_match_index]
		else:
			name = "Unknown"

		face_names.append(name)

	# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		if times != 1:
			top = int(top * (1/times))
			right = int(right * (1/times))
			bottom = int(bottom * (1/times))
			left = int(left * (1/times))

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if skip_frames:
		count += skip_frames # i.e. at 30 fps, this advances one second
		video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
