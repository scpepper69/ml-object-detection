# Gundam Face Detection Application

This application is created as client application.

Gundam Face detection is performed on the image displayed by the web camera connected to the PC.

Now, the model can detect following mobile suites.

- RX-78-2
- MS-06




## Application Architecture

Your pc should have one camera device at least.

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1h_0QSzAzVmrVb2KstaQmPc3ounfJYqUo">



## How to deploy

### Preparation

Please see README.md at parent directory.

Additional:

- TensorFlow Object Detection API

  Please see this [page](https://github.com/tensorflow/models/tree/master/research/object_detection) :


### Usage
1. docker run

   ```bash
   # GraphPipe on Windows
   cd ml-object-detection/gface/app/
   ./docker.ps1
   ```

   ```bash
   # GraphPipe on Linux
   cd ml-object-detection/gface/app/
   ./docker.sh
   ```

   ```bash
   # TensorFlow Serving on Linux
   cd ml-object-detection/gface/app/
   ./docker_tfs.sh
   ```

   

1. Startup application

   ```bash
   cd ./app
   python ./camera_gp.py
   ```

   

   Optional: For not using GraphPipe
   
   ```bash
   cd ml-object-detection/gface/app/app
   python ./camera_pb.py
   ```
   
   
   
   Optional: For using TensorFlow Serving
   
   ```bash
   cd ml-object-detection/gface/app/app
   python ./camera_tfs.py
   ```
   
   
## Application Architecture

- Learned Model : TensorFlow
- Model Server : GraphPipe or TensorFlow Serving
- Client Application : Python



## Model Structure

The model is based on ssd_mobilenet_v1_coco and fine tuned for some mobile suites faces.


