/**
 * @brief Goal
 * 1. what the PSF of a motion blur image is
 * 2. how to restore a motion blur image
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
void calcPSF(cv::Mat &outputImg, cv::Size filterSize, int len, double theta);
void fftshift(const cv::Mat &inputImg, cv::Mat &outputImg);
void filter2DFreq(const cv::Mat &inputImg, cv::Mat &outputImg, const cv::Mat &H);
void calcWnrFilter(const cv::Mat &input_h_PSF, cv::Mat &output_G, double nsr);
void edgetaper(const cv::Mat &inputImg, cv::Mat &outputImg, double gamma = 5.0, double beta = 0.2);

const char *keys =
    "{help h usage ? |             | print this message				}"
    "{image          |motion_original.jpg    | input image name	    }"
    "{LEN            |125          | length of a motion				}"
    "{THETA          |0            | angle of a motion in degrees	}"
    "{SNR            |700          | signal to noise ratio			}";

/**
 * @brief main function and the entry point of program
 */
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

    int LEN = parser.get<int>("LEN");
    double THETA = parser.get<double>("THETA");
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
    cv::imshow("Original Motional Image", imgIn);

    //! [main]
    // it needs to process even image only
    cv::Rect roi = cv::Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);

    // Hw calculation (start)
    cv::Mat Hw, h;
    calcPSF(h, roi.size(), LEN, THETA);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    // Hw calculation (stop)

    imgIn.convertTo(imgIn, CV_32F);
    edgetaper(imgIn, imgIn);

    // filtering (start)
    cv::Mat imgOut;
    filter2DFreq(imgIn(roi), imgOut, Hw);
    // filtering (stop)
    //! [main]

    imgOut.convertTo(imgOut, CV_8U);
    cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);
    cv::imshow("Deblur Motion", imgOut);
    cv::imwrite("result_motion.jpg", imgOut);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

void help()
{
    std::cout << "2023-05-06" << std::endl;
    std::cout << "Motion_deblur_v2" << std::endl;
    std::cout << "You will learn how to recover an image with motion blur distortion using a Wiener filter" << std::endl;
}

void calcPSF(cv::Mat &outputImg, cv::Size filterSize, int len, double theta)
{
    cv::Mat h(filterSize, CV_32F, cv::Scalar(0));
    cv::Point point(filterSize.width / 2, filterSize.height / 2);
    cv::ellipse(h, point, cv::Size(0, cvRound(float(len) / 2.0)),
                90.0 - theta, 0, 360, cv::Scalar(255), cv::FILLED);

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

void edgetaper(const cv::Mat &inputImg, cv::Mat &outputImg, double gamma, double beta)
{
    int Nx = inputImg.cols;
    int Ny = inputImg.rows;
    cv::Mat w1(1, Nx, CV_32F, cv::Scalar(0));
    cv::Mat w2(Ny, 1, CV_32F, cv::Scalar(0));

    float *p1 = w1.ptr<float>(0);
    float *p2 = w2.ptr<float>(0);
    float dx = float(2.0 * CV_PI / Nx);
    float x = float(-CV_PI);
    for (int i = 0; i < Nx; i++)
    {
        p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) -
                             tanh((x - gamma / 2) / beta)));
        x += dx;
    }
    float dy = float(2.0 * CV_PI / Ny);
    float y = float(-CV_PI);
    for (int i = 0; i < Ny; i++)
    {
        p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) -
                             tanh((y - gamma / 2) / beta)));
        y += dy;
    }
    cv::Mat w = w2 * w1;
    multiply(inputImg, w, outputImg);
}