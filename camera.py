#!/usr/bin/env python
import cv2
from functions import *

# if to export data
do_export = True

if do_export:
	from datetime import datetime

print("Connecting to camera...")
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
print("Camera connected...")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
	# Grab a single frame of video
	_, frame = video_capture.read()

	# Only process every other frame of video to save time
	if process_this_frame:
		times = 0.5
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=times, fy=times)
		# small_frame = frame.copy()

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]
		
		# Find all the faces and face encodings in the current frame of video
		face_locations = get_face_locations(rgb_small_frame)
		face_encodings = get_face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			# print(face_distances[best_match_index])
			if face_distances[best_match_index] <= threshold:
				name = known_face_labels[best_match_index]

				if do_export:
					export_data.append((name, datetime.now()))
			else:
				name = "Unknown"

			face_names.append(name)

	process_this_frame = not process_this_frame


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

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


if do_export:
	save_export()
