## QT and OpenCV Project

> QT 是一个跨平台应用开发框架, 使用C++编写并提供C++ API接口, 开发各种应用程序application, 尤其在graphical user interface GUI程序.

### quick start
- [QT 在线下载器阿里云镜像地址](https://developer.aliyun.com/mirror/qt)
- 利用阿里云镜像安装 QT

```shell
# downloading online.exe for QT
cd D:/DevelopTools/
# running the following command, and installing QT
.\qt-unified-windows-x64-online.exe --mirror https://mirrors.aliyun.com/qt

# TODO 注册并登录 QT 账号
# TODO 安装过程需要进行一些筛选需要安装的配件
# select the msvc2019_64_5.15.2 or msvc2019_64_6.2.4

# VS2022 install the extension QT-VS-Tools
# and then configure the QT tool in VS

# OpenCV 4.8.0
# OpenGL
```

### **Features**
- [x] Integrating QT into VS2022
- [x] QT basic widgets and Application
- [x] The plugin mechanism of Qt
- [x] The user interface (UI) by Qt Designer
- [x] Using OpenCV in QT
- [x] Using OpenGL in QT
- [ ] The Network module of Qt
- [ ] The Thread module of Qt
- [ ] The Event module of Qt
- [ ] The Database module of Qt


### Image Viewer
- Designing the user interface
    * Open an image from our hard disk 
    * Zoom in/out
    * View the previous or next image within the same folder
    * Save a copy of the current image as another file(with a different path or filename) in another format
- Reading and displaying images with QT
- Zooming in and out of images
- Saving a copy of images in any supported format
- Responding to hotkeys in a QT application
- Navigating in the folder
    * Which is the current one
    * The order in which we count them
- Responding to hotkeys
    * add a strange '&' to their text, such as &File(Alt + F) and E&xit(Alt + X)
    * Actually, this is a way of setting shortcuts in Qt.
    * Plus (+) or equal (=) for zooming in
    * Minus (-) or underscore (_) for zooming out
    * Up or left for the previous image
    * Down or right for the next image

### Editing Images Like a Professional
- Develop most of these editing features as plugins using the plugin mechanism of Qt.
- Converting images between Qt and OpenCV
- Extending an application through Qts plugin mechanism
    * each editing feature will be a plugin. 
    * Separate the same logical code and different business processing code
    * plugin mechanism to abstract a way in which can add new features easily and extensiblely
- Modifying images using image processing algorithms provided by OpenCV
- Blurring images using OpenCV
    * First, set up the UI and add the action, connect the action to a dummy slot.
    * Project [Property]->[Linker]->[System]->[SubSystem]: Console
    * Then, rewrite the dummy slot to blur the image via OpenCV library.
- omit the channel swapping between OpenCV(BGR) and QT(RGB)
    * Convert QImage to Mat, and then process the Mat and convert it back to QImage
    * All the manipulation during the processing period that we do on Mat is symmetric on the channels; that is, theres no interference between channels
    * Dont show the images during the processing period; only show them after they are converted back to QImage.
- using the plugin mechanism to add image processing function into QT gui

### Home Security Applications
- Designing and creating the user interface (UI)
- Handling cameras and videos
- Recording videos
- Calculating the FPS in real time
- Motion analysis and movement detection
- Sending notifications to a mobile in a desktop application
- Basic knowledge of multi-threading 
- Open a webcam and play the video thats been captured from it in real time
- Record video from the webcam by clicking on a start/stop button
- Show a list of saved videos
- Detect motion, save video, and send notifications to our mobile phone if suspicious motion is detected
- Show some information about the cameras and the applications status
- Network & Multi-threading in QT and Motion detection in OpenCV

### Fun with Faces

### Optical Character Recognition

### Object Detection in Real Time

### Real-Time Car Detection and Distance Measurement

### Using OpenGL for the HighSpeed Filtering of Images
- A brief introduction to OpenGL
- Using OpenGL with Qt
- Filtering images on GPU with OpenGL
- Using OpenGL with OpenCV
- OpenGL program usually involves the following steps
    * Create the context and window.
    * Prepare the data of the objects that want to draw (in 3D).
    * Pass the data to the GPU by calling some OpenGL APIs.
    * Call the drawing instructions to tell the GPU to draw the objects.
    * During the drawing, the GPU will do many manipulations on the data, and these manipulations can be customized by writing shaders in the OpenGL Shading Language.
    * Write shaders that will run on the GPU to manipulate the data on the GPU.
- the stages of the OpenGL graphics pipeline
    * Vertex shader, should write one by ourselves.
    * Shape assembly
    * Geometry shader
    * Rasterization
    * Fragment shader, should write one by ourselves.
    * Blending
- OpenCV compile flag must add support OpenGL

