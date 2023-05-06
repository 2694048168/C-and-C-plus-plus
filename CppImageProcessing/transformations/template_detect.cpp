/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use cv::GeneralizedHoughBallard and cv::GeneralizedHoughGuil
 *     to detect an object with template.
 * 
 * 该例子的模板匹配很耗时!
 *
 * @file template_detect.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */

    cv::Mat image = cv::imread("generalized_hough_mini_image.jpg");
    cv::Mat template_img = cv::imread("generalized_hough_mini_template.jpg",
                                      cv::IMREAD_GRAYSCALE);
    // TODO: the image data is loadding successfully.
    if (image.empty())
    {
        std::cout << "Could not open or find the image!\n";

        return EXIT_FAILURE;
    }
    if (template_img.empty())
    {
        std::cout << "Could not open or find the image!\n";

        return EXIT_FAILURE;
    }

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // create ballard and set options
    cv::Ptr<cv::GeneralizedHoughBallard> ballard =
        cv::createGeneralizedHoughBallard();

    ballard->setMinDist(10);
    ballard->setLevels(360);
    ballard->setDp(2);
    ballard->setMaxBufferSize(1000);
    ballard->setVotesThreshold(40);
    ballard->setCannyLowThresh(30);
    ballard->setCannyHighThresh(110);
    ballard->setTemplate(template_img);

    // create guil and set options
    cv::Ptr<cv::GeneralizedHoughGuil> guil =
        cv::createGeneralizedHoughGuil();

    guil->setMinDist(10);
    guil->setLevels(360);
    guil->setDp(3);
    guil->setMaxBufferSize(1000);
    guil->setMinAngle(0);
    guil->setMaxAngle(360);
    guil->setAngleStep(1);
    guil->setAngleThresh(1500);
    guil->setMinScale(0.5);
    guil->setMaxScale(2.0);
    guil->setScaleStep(0.05);
    guil->setScaleThresh(50);
    guil->setPosThresh(10);
    guil->setCannyLowThresh(30);
    guil->setCannyHighThresh(110);
    guil->setTemplate(template_img);

    /* create variable for location, scale and rotation of detected templates */
    std::vector<cv::Vec4f> positionBallard, positionGuil;
    int w = template_img.cols;
    int h = template_img.rows;

    // execute ballard detection
    ballard->detect(grayImage, positionBallard);
    // execute guil detection
    guil->detect(grayImage, positionGuil);

    //  draw ballard
    for (std::vector<cv::Vec4f>::iterator iter = positionBallard.begin();
         iter != positionBallard.end(); ++iter)
    {
        cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f((*iter)[0],
                                    (*iter)[1]),
                                    cv::Size2f(w * (*iter)[2], h * (*iter)[2]),
                                    (*iter)[3]);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(image, vertices[i], vertices[(i + 1) % 4],
                     cv::Scalar(255, 0, 0), 6);
    }
    //  draw guil
    for (std::vector<cv::Vec4f>::iterator iter = positionGuil.begin();
         iter != positionGuil.end(); ++iter)
    {
        cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f((*iter)[0],
                                    (*iter)[1]),
                                    cv::Size2f(w * (*iter)[2], h * (*iter)[2]),
                                    (*iter)[3]);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(image, vertices[i], vertices[(i + 1) % 4],
                     cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("result_img", image);
    cv::waitKey();

    return EXIT_SUCCESS;
}
