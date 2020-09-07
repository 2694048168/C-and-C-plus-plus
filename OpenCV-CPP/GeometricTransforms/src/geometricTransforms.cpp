/*
** function Geometric_Transforms_Demo.cpp
*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

int main(int argc, char **argv)
{
    //! [Load the image]
    // using cv::String == std::string
    // 解析命令行参数
    cv::CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");
    const cv::String filename = argc >= 2 ? parser.get<cv::String>("@input") : "./../../image/lena.jpg";
    cv::Mat src = cv::imread(filename);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n"
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }
    //! [Load the image]

    // 定义仿射变换矩阵算子
    //! [Set your 3 points to calculate the  Affine Transform]
    cv::Point2f srcTri[3];
    srcTri[0] = cv::Point2f(0.f, 0.f);
    srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);

    cv::Point2f dstTri[3];
    dstTri[0] = cv::Point2f(0.f, src.rows * 0.33f);
    dstTri[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.25f);
    dstTri[2] = cv::Point2f(src.cols * 0.15f, src.rows * 0.7f);
    //! [Set your 3 points to calculate the  Affine Transform]

    // 进行图像的几何仿射变换
    //! [Get the Affine Transform]
    cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
    //! [Get the Affine Transform]

    //! [Apply the Affine Transform just found to the src image]
    /// Set the dst image the same type and size as src
    cv::Mat warp_dst = cv::Mat::zeros(src.rows, src.cols, src.type());

    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    //! [Apply the Affine Transform just found to the src image]

    /** Rotating the image after Warp */

    //! [Compute a rotation matrix with respect to the center of the image]
    cv::Point center = cv::Point(warp_dst.cols / 2, warp_dst.rows / 2);
    double angle = -50.0;
    double scale = 0.6;
    //! [Compute a rotation matrix with respect to the center of the image]

    //! [Get the rotation matrix with the specifications above]
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
    //! [Get the rotation matrix with the specifications above]

    //! [Rotate the warped image]
    cv::Mat warp_rotate_dst;
    cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
    //! [Rotate the warped image]

    //! [Show what you got]
    cv::imshow("Source image", src);
    cv::imshow("Warp", warp_dst);
    cv::imshow("Warp + Rotate", warp_rotate_dst);
    //! [Show what you got]

    //! [Wait until user exits the program]
    cv::waitKey(0);
    //! [Wait until user exits the program]

    return 0;
}
