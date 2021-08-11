/**
 * @File    : triangulate.hpp
 * @Brief   : 用标定相机实现三维重建
 * @Author  : Wei Li
 * @Date    : 2021-08-10
*/

#ifndef TRIANGULATE
#define TRIANGULATE

#include <vector>
#include <opencv2/core.hpp>

// triangulate using Linear LS-Method
cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2);
void triangulate(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D);

#endif // TRIANGULATE