/**
 * @brief Goal
 * 1. what the gradient structure tensor is;
 * 2. how to estimate orientation and coherency of an anisotropic image
 *    by a gradient structure tensor
 * 3. how to segment an anisotropic image with a single local orientation
 *    by a gradient structure tensor
 *
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-06
 * @version Samples of OpenCV 4.7
 *
 */

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <string>

void calcGST(const cv::Mat &inputImg, cv::Mat &imgCoherencyOut,
             cv::Mat &imgOrientationOut, int w);

int main(int argc, char const *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* -------------------------------------------------- */
    const int W = 52;    // window size is WxW
    double C_Thr = 0.43; // threshold for coherency
    int LowThr = 35;     // threshold1 for orientation, it ranges from 0 to 180
    int HighThr = 57;    // threshold2 for orientation, it ranges from 0 to 180

    cv::Mat imgIn = cv::imread(cv::samples::findFile("gst_input.jpg"),
                               cv::IMREAD_GRAYSCALE);
    if (imgIn.empty()) // check whether the image is loaded or not
    {
        std::cout << "ERROR : Image cannot be loaded..!!" << std::endl;
        return -1;
    }

    //! [main_extra]
    //! [main]
    cv::Mat imgCoherency, imgOrientation;
    calcGST(imgIn, imgCoherency, imgOrientation, W);

    //! [thresholding]
    cv::Mat imgCoherencyBin;
    imgCoherencyBin = imgCoherency > C_Thr;
    cv::Mat imgOrientationBin;
    cv::inRange(imgOrientation, cv::Scalar(LowThr),
                cv::Scalar(HighThr), imgOrientationBin);
    //! [thresholding]

    //! [combining]
    cv::Mat imgBin;
    imgBin = imgCoherencyBin & imgOrientationBin;
    //! [combining]
    //! [main]

    cv::normalize(imgCoherency, imgCoherency, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(imgOrientation, imgOrientation, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imshow("Original", imgIn);
    cv::imshow("Result", 0.5 * (imgIn + imgBin));
    cv::imshow("Coherency", imgCoherency);
    cv::imshow("Orientation", imgOrientation);

    cv::imwrite("result.jpg", 0.5 * (imgIn + imgBin));
    cv::imwrite("Coherency.jpg", imgCoherency);
    cv::imwrite("Orientation.jpg", imgOrientation);

    cv::waitKey(0);
    cv::destroyAllWindows();
    //! [main_extra]

    return 0;
}

void calcGST(const cv::Mat &inputImg, cv::Mat &imgCoherencyOut,
             cv::Mat &imgOrientationOut, int w)
{
    cv::Mat img;
    inputImg.convertTo(img, CV_32F);

    // GST components calculation (start)
    // J =  (J11 J12; J12 J22) - GST
    cv::Mat imgDiffX, imgDiffY, imgDiffXY;
    cv::Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
    cv::Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
    cv::multiply(imgDiffX, imgDiffY, imgDiffXY);
    //! [calcJ_header]

    cv::Mat imgDiffXX, imgDiffYY;
    cv::multiply(imgDiffX, imgDiffX, imgDiffXX);
    cv::multiply(imgDiffY, imgDiffY, imgDiffYY);

    cv::Mat J11, J22, J12; // J11, J22 and J12 are GST components
    cv::boxFilter(imgDiffXX, J11, CV_32F, cv::Size(w, w));
    cv::boxFilter(imgDiffYY, J22, CV_32F, cv::Size(w, w));
    cv::boxFilter(imgDiffXY, J12, CV_32F, cv::Size(w, w));
    // GST components calculation (stop)

    // eigenvalue calculation (start)
    // lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    // lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    cv::Mat tmp1, tmp2, tmp3, tmp4;
    tmp1 = J11 + J22;
    tmp2 = J11 - J22;
    cv::multiply(tmp2, tmp2, tmp2);
    cv::multiply(J12, J12, tmp3);
    cv::sqrt(tmp2 + 4.0 * tmp3, tmp4);

    cv::Mat lambda1, lambda2;
    lambda1 = tmp1 + tmp4;
    lambda1 = 0.5 * lambda1; // biggest eigenvalue
    lambda2 = tmp1 - tmp4;
    lambda2 = 0.5 * lambda2; // smallest eigenvalue
    // eigenvalue calculation (stop)

    // Coherency calculation (start)
    // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    // Coherency is anisotropy degree (consistency of local orientation)
    cv::divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
    // Coherency calculation (stop)

    // orientation angle calculation (start)
    // tan(2*Alpha) = 2*J12/(J22 - J11)
    // Alpha = 0.5 atan2(2*J12/(J22 - J11))
    cv::phase(J22 - J11, 2.0 * J12, imgOrientationOut, true);
    imgOrientationOut = 0.5 * imgOrientationOut;
    // orientation angle calculation (stop)
}