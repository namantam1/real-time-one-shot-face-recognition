Realtime One Shot Face recognition using Deep Convolutional Neural Network

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
