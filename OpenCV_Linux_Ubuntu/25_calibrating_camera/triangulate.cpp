/**
 * @File    : triangulate.cpp
 * @Brief   : 用标定相机实现三维重建
 * @Author  : Wei Li
 * @Date    : 2021-08-10
*/

#include "triangulate.hpp"

// triangulate using Linear LS-Method
cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2)
{

    // system of equations assuming image=[u,v] and X=[x,y,z,1]
    // from u(p3.X)= p1.X and v(p3.X)=p2.X
    cv::Matx43d A(u1(0) * p1.at<double>(2, 0) - p1.at<double>(0, 0), u1(0) * p1.at<double>(2, 1) - p1.at<double>(0, 1), u1(0) * p1.at<double>(2, 2) - p1.at<double>(0, 2),
                  u1(1) * p1.at<double>(2, 0) - p1.at<double>(1, 0), u1(1) * p1.at<double>(2, 1) - p1.at<double>(1, 1), u1(1) * p1.at<double>(2, 2) - p1.at<double>(1, 2),
                  u2(0) * p2.at<double>(2, 0) - p2.at<double>(0, 0), u2(0) * p2.at<double>(2, 1) - p2.at<double>(0, 1), u2(0) * p2.at<double>(2, 2) - p2.at<double>(0, 2),
                  u2(1) * p2.at<double>(2, 0) - p2.at<double>(1, 0), u2(1) * p2.at<double>(2, 1) - p2.at<double>(1, 1), u2(1) * p2.at<double>(2, 2) - p2.at<double>(1, 2));

    cv::Matx41d B(p1.at<double>(0, 3) - u1(0) * p1.at<double>(2, 3),
                  p1.at<double>(1, 3) - u1(1) * p1.at<double>(2, 3),
                  p2.at<double>(0, 3) - u2(0) * p2.at<double>(2, 3),
                  p2.at<double>(1, 3) - u2(1) * p2.at<double>(2, 3));

    // X contains the 3D coordinate of the reconstructed point
    cv::Vec3d X;

    // solve AX=B
    cv::solve(A, B, X, cv::DECOMP_SVD);

    return X;
}

// triangulate a vector of image points
void triangulate(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D)
{

    for (int i = 0; i < pts1.size(); i++)
    {

        pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
    }
}