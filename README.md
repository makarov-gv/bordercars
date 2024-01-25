Problem:
----
<img align="left" style="margin-right: 12px; margin-bottom: 12px;" width="200px" src="https://downloader.disk.yandex.ru/preview/0acfca490db71932fe52b2a5416213c04d54d2b3d5afa0fe482c77b0957555b5/65b28165/FY-kbYJnFI_SYB9b0lU-KV_maoplOEewjz3u8iHBA2M6iiFSuuTcKT4lwY7DGlwKoWsW1MYPHEGnHbwYq8mS6Q%3D%3D?uid=0&filename=bordercars.jpeg&disposition=inline&hash=&limit=0&content_type=image%2Fjpeg&owner_uid=0&tknv=v2&size=2048x2048">

Ортофотопланы состоят из нескольких сшитых аэрофотоснимков, сделанных с помощью дрона. Случается такое, что во время
фотографирования проезжающая машина оказалась частично за кадром и видно лишь *одну её часть*. Но во время следующего
фотографирования машина уже проехала, и *вторая часть* в конечный ортофотоплан не попадает. Это ведет к тому, что при
создании SfM карты появляются дефектные 3D объекты.

С помощью **bordercars** детектировать и фильтровать все машины (не только обрезанные) на краях аэрофотоснимков по 
отдельности или в составе ортофотоплана в TIFF формате. Во втором случае машины будут выделяться также на стыках сшитых 
аэрофотоснимков.

На прикрепленной фотографии показан фрагмент ортофотоплана с детектированными на крае машинами. К сожалению, только 1
из 3 машин действительно обрезана (синий бокс), но подобрав корректные параметры детекции и фильтрации можно понизить
чувствительность и увеличить точность.

----

Orthophotoplanes consist of several stitched aerial photos taken with the help of a drone. While taking a photo a
passing car that is partially off-screen can occur and only *one part* of it would be visible. But while taking the next
photo, this car would have already passed, and the *second part* will not appear on final orthophotoplane. This leads to
generation of defective 3D objects when creating SfM map.

Using **bordercars** you can detect and filter out all cars (not only clipped ones) on the borders of aerial images 
individually or of whole orthophotoplane in TIFF format. In the second case, the cars will also be found at the joints 
of the stitched aerial images.

The attached image shows a fragment of an orthophotoplane with cars detected on the border. Unfortunately, only 1 out of
3 cars is really clipped (blue box), but by selecting the correct detection and filtering parameters, you can lower the
sensitivity and increase the accuracy.
<br clear="left"/>

Preparation:
----

Clone **bordercars** in your project directory:

&emsp;&emsp;```git clone https://gitverse.ru/sc/makarov/bordercars.git```

For automatic dependencies installation with CUDA support run:

&emsp;&emsp;```pip install bordercars/. && mim install mmcv-full mmdet mmrotate```

If you need to install **bordercars** manually (automatic installation fails, or you do not need CUDA), run:

1. Install PyTorch. With CUDA support:

&emsp;&emsp;```pip install torch torchvision torchaudio```

&emsp;&emsp;CPU only:

&emsp;&emsp;```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```

2. Install dependencies:

&emsp;&emsp;```pip install -U opencv-python tifffile openmim && mim install mmcv-full mmdet mmrotate```

Usage:
----

Currently, 2 models have been proven to be accurate at detecting clipped cars, both based on *Rotated FCOS (Fully
Convolutional One-Stage Object Detection)*. **rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90** is theoretically better
when processing bigger images (tested on 10000x10000), whereas **rotated_fcos_r50_fpn_1x_dota_le90** better processes
smaller images (tested on 2000x2000). Their PyTorch model files are located in *models* folder, whereas their python
config files are located in *configs* folder.

Import border cars detector class with ```from bordercars.detector import Detector``` and use its PEP8 documented
methods which include orthophotoplane inference (on class object *call*), bounding box manipulations and **draw** method
that allows to draw bounding boxes on top of orthophotoplane and save it for debugging purposes. When initializing
*Detector* class object, pass PyTorch model file and its corresponding python config file from preinstalled choices or
of your own.

Import border cars detector test utility with ```from bordercars.test_detector import test``` and use this function for
testing purposes. It processes **test.tif** orthophotoplane (either yours or an example from Yandex Disk that will be
downloaded automatically if not found) with chosen model. After that, bounding boxes list as a text file and a copy of
orthophotoplane with drawn bounding boxes are saved to *results* folder in your project directory.