#!/usr/bin/env python
from sys import argv

if len(argv) == 1:
    print(f"Usage: {argv[0]} <image file path>")
    exit(1)


IMG_FILE = argv[1]


from functions import *


rgb_img = load_image_file(IMG_FILE)

# Find all the faces and face encodings in the current frame of video
face_encodings = get_face_encodings(rgb_img)

print("============================================")
print(f"Total faces found: {len(face_encodings)}")

for face_encoding in face_encodings:
    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    print(
        f"{face_distances[best_match_index]<=threshold} ({known_face_labels[best_match_index]}): "
        f"Distance {face_distances[best_match_index]}, "
        f"Threshold {threshold}"
    )

print("============================================")
