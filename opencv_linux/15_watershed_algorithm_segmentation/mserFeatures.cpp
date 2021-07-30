/**
 * @File    : mserFeatures.cpp
 * @Brief   : 最大稳定外部区域（MSER）算法从图像中提取有意义的区域
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

int main(int argc, char **argv)
{
    // Read input image
    cv::Mat image = cv::imread("./../images/building.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading building image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Image");
    cv::imshow("Image", image);

    // basic MSER detector
    cv::Ptr<cv::MSER> ptrMSER = cv::MSER::create(5,     // delta value for local minima detection
                                                 200,   // min acceptable area
                                                 2000); // max acceptable area

    // vector of point sets
    std::vector<std::vector<cv::Point>> points;
    // vector of rectangles
    std::vector<cv::Rect> rects;
    // detect MSER features
    ptrMSER->detectRegions(image, points, rects);

    std::cout << points.size() << " MSERs detected" << std::endl;

    // create white image
    cv::Mat output(image.size(), CV_8UC3);
    output = cv::Scalar(255, 255, 255);

    // OpenCV random number generator
    cv::RNG rng;

    // Display the MSERs in color areas
    // for each detected feature
    // reverse order to display the larger MSER first
    for (std::vector<std::vector<cv::Point>>::reverse_iterator it = points.rbegin(); it != points.rend(); ++it)
    {
        // generate a random color
        cv::Vec3b c(rng.uniform(0, 254),
                    rng.uniform(0, 254),
                    rng.uniform(0, 254));

        std::cout << "MSER size= " << it->size() << std::endl;

        // for each point in MSER set
        for (std::vector<cv::Point>::iterator itPts = it->begin(); itPts != it->end(); ++itPts)
        {
            //do not overwrite MSER pixels
            if (output.at<cv::Vec3b>(*itPts)[0] == 255)
            {
                output.at<cv::Vec3b>(*itPts) = c;
            }
        }
    }

    cv::namedWindow("MSER point sets");
    cv::imshow("MSER point sets", output);
    cv::imwrite("mser.bmp", output);

    // Extract and display the rectangular MSERs
    std::vector<cv::Rect>::iterator itr = rects.begin();
    std::vector<std::vector<cv::Point>>::iterator itp = points.begin();
    for (; itr != rects.end(); ++itr, ++itp)
    {
        // ratio test
        if (static_cast<double>(itp->size()) / itr->area() > 0.6)
            cv::rectangle(image, *itr, cv::Scalar(255), 2);
    }

    // Display the resulting image
    cv::namedWindow("Rectangular MSERs");
    cv::imshow("Rectangular MSERs", image);

    // Reload the input image
    image = cv::imread("./../images/building.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading building image file." << std::endl;
        return 1;
    }

    // Extract and display the elliptic MSERs
    for (std::vector<std::vector<cv::Point>>::iterator it = points.begin(); it != points.end(); ++it)
    {
        // for each point in MSER set
        for (std::vector<cv::Point>::iterator itPts = it->begin(); itPts != it->end(); ++itPts)
        {
            // Extract bouding rectangles
            cv::RotatedRect rr = cv::minAreaRect(*it);
            // check ellipse elongation
            if (rr.size.height / rr.size.height > 0.6 || rr.size.height / rr.size.height < 1.6)
                cv::ellipse(image, rr, cv::Scalar(255), 2);
        }
    }

    // Display the image
    cv::namedWindow("MSER ellipses");
    cv::imshow("MSER ellipses", image);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
