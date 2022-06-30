import os
import numpy as np
from PIL import Image
import dlib


# BASE = "C:/Users/Naman Tamrakar/Desktop/ML-CCPD/real-time-one-shot-face-recognition/"
BASE = ""

file_name = BASE + "encodings/database.npz"
changed = False
print(f"Database file: {file_name}")

# metrix = "cosine"
# threshold = 0.06
metrix = "euclidean"
threshold = 0.5
print(f"Metrix funtion is {metrix} and threshold {threshold}")

face_detector = dlib.get_frontal_face_detector()
print("Face detector model loded...")


predictor_model = BASE + "models/shape_predictor_5_face_landmarks.dat"
pose_predictor = dlib.shape_predictor(predictor_model)
print("Face landmarks model loded...")

face_recognition_model = BASE + "models/dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
print("Face recognition model loded...")

EXPORT_FILE = "data.csv"
export_data = []

try:
    known_face_encodings, known_face_labels = np.load(file_name).values()
except IOError:
	known_face_encodings, known_face_labels = np.array([]), np.array([], "str")
	changed = True


def save_data():
	np.savez(file_name, known_face_encodings, known_face_labels)


def save_export():
	from pandas import DataFrame, concat

	df = DataFrame(export_data, columns=["Name", "Time"])
	df = concat([
		df.drop_duplicates(subset=["Name"], keep='first'),
		df.drop_duplicates(subset=["Name"], keep='last'),
	])
	df.to_csv(EXPORT_FILE, index=False)
	print(f"Data saved to - `{EXPORT_FILE}`")


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
		face_locations = face_detector(face_image, 1)
	else:
		face_locations = [css_to_rect(face_location) for face_location in face_locations]

	return [pose_predictor(face_image, face_location) for face_location in face_locations]


def get_face_encodings(face_image, known_face_locations=None, num_jitters=1):
	raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
	return [np.array(face_encoder.compute_face_descriptor(
		face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def get_face_locations(img):
	return [trim_css_to_bounds(rect_to_css(face), img.shape) for face in face_detector(img, 1)]

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
	# else:
	# 	print(f"Image `{label}` already exist with same name")

def remove_image(label):
	global known_face_labels, known_face_encodings, changed

	known_face_encodings = known_face_encodings[known_face_labels != label]
	known_face_labels = known_face_labels[known_face_labels != label]
	changed = True

## add images
for dir, _, files in os.walk("images"):
	for file in files:
		add_image(os.path.join(dir, file))

# saving updated encoding on close
if changed:
	save_data()

print(f"Total Faces in database: {len(known_face_labels)}")
print(f"Encoding shape in database: {known_face_encodings.shape}")
