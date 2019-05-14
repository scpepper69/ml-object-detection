# TensorFlow Object Detection Tutorial Example Application

This application is created as client application.

Object detection is performed on the image displayed by the web camera connected to the PC.




## Application Architecture

your pc should have one camera device at least.

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1h_0QSzAzVmrVb2KstaQmPc3ounfJYqUo">



## How to deploy

### Preparation

Please see README.md at parent directory.




### Usage
1. docker run (windows powershell)

   ```bash
   cd ml-object-detection/tf_tutorial/app/
   ./docker.ps1
   ```

1. Startup application

   ```bash
   cd ./app
   python ./camera_gp.py
   ```

   

   Optional: For not using GraphPipe

   ```bash
   cd ml-object-detection/tf_tutorial/app/app
   python ./camera_pb.py
   ```



## Application Architecture

- Learned Model : TensorFlow
- Model Server : GraphPipe
- Client Application : Python



## Model Structure

Model file is not included.

I tried some models published at [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and checked working well.

