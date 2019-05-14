# Gundam Face Detection Application

This application is created as client application.

Gundam Face detection is performed on the image displayed on the web camera connected to the PC.

Now, the model can detect RX-78-2 and MS-06.




## Application Architecture

your pc should have one camera device at least.

<img class="aligncenter size-full" src="https://drive.google.com/uc?export=view&id=1h_0QSzAzVmrVb2KstaQmPc3ounfJYqUo">



## How to deploy

### Preparation

Please see README.md at parent directory.

### Usage
1. docker run (windows powershell)

   ```bash
   # cd ml-object-detection/gface/app/
   # ./docker.ps1
   ```

1. Startup hiragana.py

   ```bash
   # cd ./app
   # python ./camera_gp.py
   ```
   
   
## Application Architecture

- Learned Model : TensorFlow
- Model Server : GraphPipe
- Client Application : Python



## Model Structure

The model is based on ssd_mobilenet_v1_coco and fine tuned for detecting gundam faces.

