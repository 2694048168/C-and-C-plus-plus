/**
 * @brief Goal
 * 1. what a degradation image model is
 * 2. what the PSF of an out-of-focus image is
 * 3. how to restore a blurred image
 * 4. what is a Wiener filter
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

void help();
void calcPSF(cv::Mat &outputImg, cv::Size filterSize, int R);
void fftshift(const cv::Mat &inputImg, cv::Mat &outputImg);
void filter2DFreq(const cv::Mat &inputImg, cv::Mat &outputImg, const cv::Mat &H);
void calcWnrFilter(const cv::Mat &input_h_PSF, cv::Mat &output_G, double nsr);

const cv::String keys =
    "{help h usage ? |             | print this message   }"
    "{image          |original.jpg | input image name     }"
    "{R              |5           | radius               }"
    "{SNR            |100         | signal to noise ratio}";

int main(int argc, char *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* -------------------------------------------------- */
    help();
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int R = parser.get<int>("R");
    int snr = parser.get<int>("SNR");
    std::string strInFileName = parser.get<cv::String>("image");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    cv::Mat imgIn = cv::imread(cv::samples::findFile(strInFileName),
                               cv::IMREAD_GRAYSCALE);
    if (imgIn.empty()) // check whether the image is loaded or not
    {
        std::cout << "ERROR : Image cannot be loaded..!!" << std::endl;
        return -1;
    }

    //! [main]
    // it needs to process even image only
    cv::Rect roi = cv::Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);

    // Hw calculation (start)
    cv::Mat Hw, h;
    calcPSF(h, roi.size(), R);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    // Hw calculation (stop)

    // filtering (start)
    cv::Mat imgOut;
    filter2DFreq(imgIn(roi), imgOut, Hw);
    // filtering (stop)
    //! [main]

    imgOut.convertTo(imgOut, CV_8U);
    cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);
    cv::imshow("Original", imgIn);
    cv::imshow("Debluring", imgOut);
    cv::imwrite("result.jpg", imgOut);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

void help()
{
    std::cout << "2023-05-06\n";
    std::cout << "DeBlur_v8\n";
    std::cout << "You will learn how to recover an out-of-focus image by Wiener filter" << std::endl;
}

void calcPSF(cv::Mat &outputImg, cv::Size filterSize, int R)
{
    cv::Mat h(filterSize, CV_32F, cv::Scalar(0));
    cv::Point point(filterSize.width / 2, filterSize.height / 2);
    cv::circle(h, point, R, 255, -1, 8);
    cv::Scalar summa = sum(h);
    outputImg = h / summa[0];
}

void fftshift(const cv::Mat &inputImg, cv::Mat &outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    cv::Mat q0(outputImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(outputImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(outputImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(outputImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void filter2DFreq(const cv::Mat &inputImg, cv::Mat &outputImg, const cv::Mat &H)
{
    cv::Mat planes[2] = {cv::Mat_<float>(inputImg.clone()),
                         cv::Mat::zeros(inputImg.size(), CV_32F)};

    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI, cv::DFT_SCALE);

    cv::Mat planesH[2] = {cv::Mat_<float>(H.clone()),
                          cv::Mat::zeros(H.size(), CV_32F)};
    cv::Mat complexH;
    cv::merge(planesH, 2, complexH);
    cv::Mat complexIH;
    cv::mulSpectrums(complexI, complexH, complexIH, 0);

    cv::idft(complexIH, complexIH);
    cv::split(complexIH, planes);
    outputImg = planes[0];
}

void calcWnrFilter(const cv::Mat &input_h_PSF, cv::Mat &output_G, double nsr)
{
    cv::Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    cv::Mat planes[2] = {cv::Mat_<float>(h_PSF_shifted.clone()),
                         cv::Mat::zeros(h_PSF_shifted.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes);
    cv::Mat denom;
    cv::pow(abs(planes[0]), 2, denom);
    denom += nsr;
    cv::divide(planes[0], denom, output_G);
}

