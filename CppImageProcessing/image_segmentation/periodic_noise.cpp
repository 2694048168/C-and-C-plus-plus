/**
 * @brief Goal
 * 1. how to remove periodic noise in the Fourier domai
 *
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-06
 * @version Samples of OpenCV 4.7
 *
 */

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

void fftshift(const cv::Mat &inputImg, cv::Mat &outputImg);
void filter2DFreq(const cv::Mat &inputImg, cv::Mat &outputImg, const cv::Mat &H);
void synthesizeFilterH(cv::Mat &inputOutput_H, cv::Point center, int radius);
void calcPSD(const cv::Mat &inputImg, cv::Mat &outputImg, int flag = 0);

const cv::String keys =
    "{help h usage ? |             | print this message   }"
    "{@image          |period_input.jpg | input image name     }";

/**
 * @brief main function for the entry point
 */
int main(int argc, char *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* -------------------------------------------------- */
    cv::CommandLineParser parser(argc, argv, keys);
    cv::String strInFileName = parser.get<cv::String>("@image");
    cv::Mat imgIn = cv::imread(cv::samples::findFile(strInFileName),
                               cv::IMREAD_GRAYSCALE);
    if (imgIn.empty()) // check whether the image is loaded or not
    {
        std::cout << "ERROR : Image cannot be loaded..!!" << std::endl;
        return -1;
    }

    cv::imshow("Image corrupted", imgIn);
    imgIn.convertTo(imgIn, CV_32F);

    //! [main]
    // it needs to process even image only
    cv::Rect roi = cv::Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
    imgIn = imgIn(roi);

    // PSD calculation (start)
    cv::Mat imgPSD;
    calcPSD(imgIn, imgPSD);
    fftshift(imgPSD, imgPSD);
    cv::normalize(imgPSD, imgPSD, 0, 255, cv::NORM_MINMAX);
    // PSD calculation (stop)

    // H calculation (start)
    cv::Mat H = cv::Mat(roi.size(), CV_32F, cv::Scalar(1));
    const int r = 21;
    synthesizeFilterH(H, cv::Point(705, 458), r);
    synthesizeFilterH(H, cv::Point(850, 391), r);
    synthesizeFilterH(H, cv::Point(993, 325), r);
    // H calculation (stop)
    //  filtering (start)
    cv::Mat imgOut;
    fftshift(H, H);
    filter2DFreq(imgIn, imgOut, H);
    // filtering (stop)
    //! [main]

    imgOut.convertTo(imgOut, CV_8U);
    cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);

    cv::imwrite("result.jpg", imgOut);
    cv::imwrite("PSD.jpg", imgPSD);

    fftshift(H, H);
    cv::normalize(H, H, 0, 255, cv::NORM_MINMAX);
    cv::imshow("Debluring", imgOut);
    cv::imwrite("filter.jpg", H);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
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

void synthesizeFilterH(cv::Mat &inputOutput_H, cv::Point center, int radius)
{
    cv::Point c2 = center, c3 = center, c4 = center;
    c2.y = inputOutput_H.rows - center.y;
    c3.x = inputOutput_H.cols - center.x;
    c4 = cv::Point(c3.x, c2.y);
    cv::circle(inputOutput_H, center, radius, 0, -1, 8);
    cv::circle(inputOutput_H, c2, radius, 0, -1, 8);
    cv::circle(inputOutput_H, c3, radius, 0, -1, 8);
    cv::circle(inputOutput_H, c4, radius, 0, -1, 8);
}


// Function calculates PSD(Power spectrum density) by fft with two flags
// flag = 0 means to return PSD
// flag = 1 means to return log(PSD)
//! [calcPSD]
void calcPSD(const cv::Mat &inputImg, cv::Mat &outputImg, int flag)
{
    cv::Mat planes[2] = {cv::Mat_<float>(inputImg.clone()),
                         cv::Mat::zeros(inputImg.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    planes[0].at<float>(0) = 0;
    planes[1].at<float>(0) = 0;

    // compute the PSD = sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)^2
    cv::Mat imgPSD;
    cv::magnitude(planes[0], planes[1], imgPSD); // imgPSD = sqrt(Power spectrum density)
    cv::pow(imgPSD, 2, imgPSD);                  // it needs ^2 in order to get PSD
    outputImg = imgPSD;

    // logPSD = log(1 + PSD)
    if (flag)
    {
        cv::Mat imglogPSD;
        imglogPSD = imgPSD + cv::Scalar::all(1);
        cv::log(imglogPSD, imglogPSD);
        outputImg = imglogPSD;
    }
}