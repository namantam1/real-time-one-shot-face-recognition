#!/usr/bin/env python
import os
import numpy as np
from PIL import Image
import dlib

file_name = "encodings/database.npz"
changed = False

face_recognition_model = "models/dlib_face_recognition_resnet_model_v1.dat"
face_encoder = None

face_detector = None

predictor_model = "models/shape_predictor_5_face_landmarks.dat"
pose_predictor = None

try:
    known_face_encodings, known_face_labels = np.load(file_name).values()
except IOError:
	known_face_encodings, known_face_labels = np.array([]), np.array([], "str")
	changed = True
# print("loaded...")

def save_data():
	np.savez(file_name, known_face_encodings, known_face_labels)

def get_face_encoder():
	global face_encoder
	if face_encoder is None:
		face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
	return face_encoder

def get_face_detector():
	global face_detector

	if face_detector is None:
		# can use cnn detector also default is hog(fast)
		face_detector = dlib.get_frontal_face_detector()
	return face_detector

def get_pose_predictor():
	global pose_predictor
	if pose_predictor is None:
		pose_predictor = dlib.shape_predictor(predictor_model)
	return pose_predictor

def load_image_file(file):
	im = Image.open(file)
	im = im.convert("RGB")
	return np.array(im)

def css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

metrix = "cosine"
threshold = 0.07
def face_distance(encodings, encoding):
    if len(encodings) == 0:
        return np.empty(0)

    if metrix == "euclidean":
        return np.linalg.norm(encodings - encoding, axis=1)
    else:
        a1 = np.sum(np.multiply(encodings, encoding), axis=1)
        b1 = np.sum(np.multiply(encodings, encodings), axis=1)
        c1 = np.sum(np.multiply([encoding], [encoding]), axis=1)
        return (1 - (a1 / (b1**.5 * c1**.5)))

def _raw_face_landmarks(face_image, face_locations=None):
	if face_locations is None:
		face_locations = get_face_detector()(face_image, 1)
	else:
		face_locations = [css_to_rect(face_location) for face_location in face_locations]

	return [get_pose_predictor()(face_image, face_location) for face_location in face_locations]


def get_face_encodings(face_image, known_face_locations=None, num_jitters=1):
	raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
	return [np.array(get_face_encoder().compute_face_descriptor(
		face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def get_face_locations(img):
	return [trim_css_to_bounds(rect_to_css(face), img.shape) for face in get_face_detector()(img, 1)]

def add_image(image_path):
	global known_face_labels, known_face_encodings, changed

	root, _ = os.path.splitext(image_path)
	label = os.path.split(root)[-1]

	if not np.isin(label, known_face_labels):
		print(f"Adding {label} ...")
		image = load_image_file(image_path)
		image_encoding = get_face_encodings(image)[0]

		if known_face_labels.size == 0:
			known_face_encodings = np.array([image_encoding])
			known_face_labels = np.array([label])
		else:
			known_face_encodings = np.vstack([known_face_encodings, image_encoding])
			known_face_labels = np.append(known_face_labels, label)
		print(f"Added {label}")
		
		changed = True
	else:
		print(f"Image `{label}` already exist with same name")

def remove_image(label):
	global known_face_labels, known_face_encodings, changed

	known_face_encodings = known_face_encodings[known_face_labels != label]
	known_face_labels = known_face_labels[known_face_labels != label]
	changed = True


def main():
	import cv2
	from urllib import request

    # uncomment below code to connect to remote camera
	class video_capture:
		""" Class to connect to remote camera """
		url = "http://192.168.137.155:8080/shot.jpg"
		@staticmethod
		def read():
			imgResp = request.urlopen(__class__.url)
			imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
			img = cv2.imdecode(imgNp, -1)
			# img = cv2.resize(img, (640, 480)) # use this if size recieve is very large
			return None, img
		@staticmethod
		def release():
			pass

	# Get a reference to webcam #0 (the default one)
	# video_capture = cv2.VideoCapture(0)

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
				# if matches[best_match_index]:
				if face_distances[best_match_index] < threshold:
					name = known_face_labels[best_match_index]
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

## add images
for dir, _, files in os.walk("images"):
	for file in files:
		add_image(os.path.join(dir, file))

# add_image("application_data/images/ravi sir.jpg")
print(known_face_labels, known_face_encodings.shape)

# running server
main()

# saving updated encoding on close
if changed:
	save_data()