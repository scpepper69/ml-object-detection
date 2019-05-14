# Face Counter Application

This application is created as client application.

Face detection is performed on the image displayed by the web camera connected to the PC.




## Application Architecture

your pc should have one camera device at least.

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1h_0QSzAzVmrVb2KstaQmPc3ounfJYqUo">



## How to deploy

### Preparation

Please see README.md at parent directory.




### Usage
1. docker run (windows powershell)

   ```bash
   cd ml-object-detection/human_face/app/
   ./docker.ps1
   ```

1. Startup application

   ```bash
   cd ./app
   python ./camera_gp.py
   ```

   

   Optional: For not using GraphPipe

   ```bash
   cd ml-object-detection/human_face/app/app
   python ./camera_pb.py
   ```



## Application Architecture

- Learned Model : TensorFlow
- Model Server : GraphPipe
- Client Application : Python



## Model Structure

Model file is not included.

You can use some face detection models like [this](https://github.com/the-house-of-black-and-white/hall-of-faces).

