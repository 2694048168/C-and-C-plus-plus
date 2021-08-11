/**
 * @File    : detectObjects.cpp
 * @Brief   : 用 Haar 级联实现人脸检测
 *     OpenCV 中有一些预先训练好的级联分类器，可用于检测人脸、脸部特征、人类和其他物体
 *     这些级联分类器以 XML 文件的形式存储在源文件的 data 目录下
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

// ------------------------------
int main(int argc, char **argv)
{
    // open the positive sample images
    std::vector<cv::Mat> referenceImages;

    cv::Mat stop00 = cv::imread("./../images/stopSamples/stop00.png");
    cv::Mat stop01 = cv::imread("./../images/stopSamples/stop01.png");
    cv::Mat stop02 = cv::imread("./../images/stopSamples/stop02.png");
    cv::Mat stop03 = cv::imread("./../images/stopSamples/stop03.png");
    cv::Mat stop04 = cv::imread("./../images/stopSamples/stop04.png");
    cv::Mat stop05 = cv::imread("./../images/stopSamples/stop05.png");
    cv::Mat stop06 = cv::imread("./../images/stopSamples/stop06.png");
    cv::Mat stop07 = cv::imread("./../images/stopSamples/stop07.png");
    if (!stop01.data || !stop02.data || !stop03.data || !stop04.data || !stop05.data || !stop06.data || !stop07.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }

    referenceImages.push_back(stop00);
    referenceImages.push_back(stop01);
    referenceImages.push_back(stop02);
    referenceImages.push_back(stop03);
    referenceImages.push_back(stop04);
    referenceImages.push_back(stop05);
    referenceImages.push_back(stop06);
    referenceImages.push_back(stop07);

    // create a composite image
    cv::Mat positveImages(2 * referenceImages[0].rows, 4 * referenceImages[0].cols, CV_8UC3);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
        {
            referenceImages[i * 2 + j].copyTo(positveImages(cv::Rect(j * referenceImages[i * 2 + j].cols, i * referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
        }

    cv::imshow("Positive samples", positveImages);

    cv::Mat negative = cv::imread("./../images/stopSamples/bg01.jpg");
    cv::resize(negative, negative, cv::Size(), 0.33, 0.33);
    cv::imshow("One negative sample", negative);

    cv::Mat inputImage = cv::imread("./../images/stopSamples/stop9.jpg");
    cv::resize(inputImage, inputImage, cv::Size(), 0.5, 0.5);

    cv::CascadeClassifier cascade;
    if (!cascade.load("./../images/stopSamples/classifier/cascade.xml"))
    {
        std::cout << "Error when loading the cascade classfier!" << std::endl;
        return -1;
    }

    // predict the label of this image
    std::vector<cv::Rect> detections;

    cascade.detectMultiScale(inputImage,          // input image
                             detections,          // detection results
                             1.1,                 // scale reduction factor
                             1,                   // number of required neighbor detections
                             0,                   // flags (not used)
                             cv::Size(48, 48),    // minimum object size to be detected
                             cv::Size(128, 128)); // maximum object size to be detected

    std::cout << "detections= " << detections.size() << std::endl;
    for (int i = 0; i < detections.size(); i++)
        cv::rectangle(inputImage, detections[i], cv::Scalar(255, 255, 255), 2);

    cv::imshow("Stop sign detection", inputImage);

    // Detecting faces
    cv::Mat picture = cv::imread("./../images/girl.jpg");
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("./../images/haarcascades/haarcascade_frontalface_default.xml"))
    {
        std::cout << "Error when loading the face cascade classfier!" << std::endl;
        return -1;
    }

    faceCascade.detectMultiScale(picture,             // input image
                                 detections,          // detection results
                                 1.1,                 // scale reduction factor
                                 3,                   // number of required neighbor detections
                                 0,                   // flags (not used)
                                 cv::Size(48, 48),    // minimum object size to be detected
                                 cv::Size(128, 128)); // maximum object size to be detected

    std::cout << "detections= " << detections.size() << std::endl;
    // draw detections on image
    for (int i = 0; i < detections.size(); i++)
        cv::rectangle(picture, detections[i], cv::Scalar(255, 255, 255), 2);

    // Detecting eyes
    cv::CascadeClassifier eyeCascade;
    if (!eyeCascade.load("./../images/haarcascades/haarcascade_eye.xml"))
    {
        std::cout << "Error when loading the eye cascade classfier!" << std::endl;
        return -1;
    }

    eyeCascade.detectMultiScale(picture,           // input image
                                detections,        // detection results
                                1.1,               // scale reduction factor
                                3,                 // number of required neighbor detections
                                0,                 // flags (not used)
                                cv::Size(24, 24),  // minimum object size to be detected
                                cv::Size(64, 64)); // maximum object size to be detected

    std::cout << "detections= " << detections.size() << std::endl;
    // draw detections on image
    for (int i = 0; i < detections.size(); i++)
        cv::rectangle(picture, detections[i], cv::Scalar(0, 0, 0), 2);

    cv::imshow("Detection results", picture);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
