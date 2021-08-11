/**
 * @File    : recognizeFace.cpp
 * @Brief   : 基于局部二值模式（local binary pattern， LBP） 的人脸识别
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>

// compute the Local Binary Patterns of a gray-level image
void lbp(const cv::Mat &image, cv::Mat &result)
{
    assert(image.channels() == 1); // input image must be gray scale

    result.create(image.size(), CV_8U); // allocate if necessary

    for (int j = 1; j < image.rows - 1; j++)
    { // for all rows (except first and last)

        const uchar *previous = image.ptr<const uchar>(j - 1); // previous row
        const uchar *current = image.ptr<const uchar>(j);      // current row
        const uchar *next = image.ptr<const uchar>(j + 1);     // next row

        uchar *output = result.ptr<uchar>(j); // output row

        for (int i = 1; i < image.cols - 1; i++)
        {
            // compose local binary pattern
            *output = previous[i - 1] > current[i] ? 1 : 0;
            *output |= previous[i] > current[i] ? 2 : 0;
            *output |= previous[i + 1] > current[i] ? 4 : 0;

            *output |= current[i - 1] > current[i] ? 8 : 0;
            *output |= current[i + 1] > current[i] ? 16 : 0;

            *output |= next[i - 1] > current[i] ? 32 : 0;
            *output |= next[i] > current[i] ? 64 : 0;
            *output |= next[i + 1] > current[i] ? 128 : 0;

            output++; // next pixel
        }
    }

    // Set the unprocess pixels to 0
    result.row(0).setTo(cv::Scalar(0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0));
    result.col(result.cols - 1).setTo(cv::Scalar(0));
}

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = imread("./../images/girl.jpg", cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    cv::imshow("Original image", image);

    cv::Mat lbpImage;
    lbp(image, lbpImage);
    cv::imshow("LBP image", lbpImage);

    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer =
        cv::face::LBPHFaceRecognizer::create(1,     // radius of LBP pattern
                                             8,     // the number of neighboring pixels to consider
                                             8, 8,  // grid size
                                             200.); // minimum distance to nearest neighbor

    // vectors of reference image and their labels
    std::vector<cv::Mat> referenceImages;
    std::vector<int> labels;
    // open the reference images
    cv::Mat face0_1 = imread("./../images/face0_1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat face0_2 = imread("./../images/face0_2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat face1_1 = imread("./../images/face1_1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat face1_2 = imread("./../images/face1_2.png", cv::IMREAD_GRAYSCALE);
    if (!face0_1.data || !face0_2.data || !face1_1.data || !face1_2.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    referenceImages.push_back(face0_1);
    labels.push_back(0); // person 0
    referenceImages.push_back(face0_2);
    labels.push_back(0); // person 0
    referenceImages.push_back(face1_1);
    labels.push_back(1); // person 1
    referenceImages.push_back(face1_2);
    labels.push_back(1); // person 1

    // the 4 positive samples
    cv::Mat faceImages(2 * referenceImages[0].rows, 2 * referenceImages[0].cols, CV_8U);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {

            referenceImages[i * 2 + j].copyTo(faceImages(cv::Rect(j * referenceImages[i * 2 + j].cols, i * referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
        }

    cv::resize(faceImages, faceImages, cv::Size(), 0.5, 0.5);
    cv::imshow("Reference faces", faceImages);

    // train the recognizer by
    // computing the LBPHs
    recognizer->train(referenceImages, labels);

    int predictedLabel = -1;
    double confidence = 0.0;

    // Extract a face image
    cv::Mat inputImage;
    cv::resize(image(cv::Rect(160, 75, 90, 90)), inputImage, cv::Size(256, 256));
    cv::imshow("Input image", inputImage);

    // predict the label of this image
    recognizer->predict(inputImage,     // face image
                        predictedLabel, // predicted label of this image
                        confidence);    // confidence of the prediction

    std::cout << "Image label= " << predictedLabel << " (" << confidence << ")" << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
