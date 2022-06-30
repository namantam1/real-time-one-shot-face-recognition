Realtime One Shot Face recognition using Deep Convolutional Neural Network

# Introduction

The task is to detect whether a face is present in database or not.

This seems very straight forward, Train a *Convolutional Neural Network* model using labelled faces present in database. 
(label may be name, regitration no., mobile or email). and  use this model to identify those faces.

### But! What is problem with this method?

Suppose a new person joins or leaves the organisation then we will have to retrain our CNN model with updated labelled faces. If we retrain the model again
 it will consume time and computational resources again to train model which is not feasible for big organisation.
 
### How to takle this problem?

To overcome this problem we use a one-shot face recognition CNN model in which we train it in such a way it will check whether two images are of same
 person or not. So model will compare the detected face in with all faces in database.
 
So whenever a person joins or leaves the organisation we just have to update our database.
 
# Steps involves in One-shot face recognition

1. Face detection
2. Posing and projecting face
3. Face regonition (Face matching)

## Face Detection

Detecting the face using one of face detection algorith i.e. **Viola Jones Algorithm**, **Histogram of Oriented Gradients** and cropped out the deteted faces in
 a video frame or image.

<img src="https://user-images.githubusercontent.com/59503233/176749915-c1ab36f3-eb07-4612-8ac4-b69b639c4242.png" height="220" />

## Posing and projecting face

If the detected face is somewhat rotated or tilted, then we try to align the face in frontal pose using **Facial landmarks**. This improves the face recognition
 accuracy very well.

<img src="https://user-images.githubusercontent.com/59503233/176746145-04c0d5ae-cae5-4226-8ff2-97233a33caab.png" height="200" />

**Facial Landmarks**

<img src="https://user-images.githubusercontent.com/59503233/176747942-3e186443-57e9-4efe-b805-f981ba1c9bff.png" width="70%" />

**Face Alignment**

## Face Matching

In this final step we use a DCNN model like Resnet50, Facenet, mobilenet, Alexnet, VGG net, etc to encode preprocessed image into a 128 size floating vector.
 This 128 size floating vector is use the calculate the **Euclidean** or **Cosine** distance between two images. If the distance is below a threshold limit we
 assume those images of same person. 
 

<img src="https://user-images.githubusercontent.com/59503233/176750782-d85d2319-56f6-404f-950e-0cd7a52d9529.png" height="350" />

**128 size encoded vector from Image**

# Setup

## Prerequisite

Please check `requirements.txt` for version of packages.

1. Python >=3.7.0
2. Numpy
3. Dlib
4. OpenCV
5. Pillow
6. Tensorflow (For Notebook only)

> NOTE: To install `Dlib` on windows search `how to install dlib on windows` and follow instructions.

## How to run?

1. Just put all the images in `images` folder from which you want to identify faces. (The name of image will be used as label without extension)
2. Now Run the following command to start the app.

```bash
python main.py
```

4. Now a new window will open where you can see yourself.
5. To close that window press `q`.

## How to use mobile camera?

1. Download `IP Webcam` app on you mobile device.
2. Now in setting of app decrease the resolution to `640x480` and quality to `50` for better experience.
3. Now start the server in app.
4. An IP will be show on screen like `http://192.168.***.***:8080`, Copy this and paste it on `main.py`
 line `139` along like this `http://192.168.***.***:8080/shot.jpg` and uncomment `video_capture` class.
5. Now run the python command given above to connnect to mobile camera.

# Demo of real time face recognition
![demo-video](https://user-images.githubusercontent.com/59503233/175144109-b207a910-87d6-44ac-8d36-b175910d2683.gif)

# References

* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://ieeexplore.ieee.org/document/7298682) (Research Paper)
* [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) (Research Paper)
* [One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d) (Blog)
* [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) (Blog)
* [Transfer Learning by Codebasic Channel](https://www.youtube.com/watch?v=LsdxvjLWkIY&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=27)  (Youtube)
* [Build Face recognition App - Paper2Code by Nicholas Renotte](https://www.youtube.com/watch?v=bK_k7eebGgc) (Youtube)
