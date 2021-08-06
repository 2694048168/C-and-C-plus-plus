/**
 * @File    : watershedSegmentation.hpp
 * @Brief   : 用分水岭算法实现图像分割
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

/**分水岭变换是一种流行的图像处理算法，用于快速将图像分割成多个同质区域。
 * 它基于这样的思想：如果把图像看作一个拓扑地貌，那么同类区域就相当于陡峭边缘内相对平坦的盆地。
 * 分水岭算法通过逐步增高水位，把地貌分割成多个部分。
 * 因为算法很简单，它的原始版本会过度分割图像，产生很多小的区域。
 * 因此 OpenCV 提出了该算法的改进版本，使用一系列预定义标记来引导图像分割的定义方式。
 */

#ifndef WATERSHED_ALGORITHM
#define WATERSHED_ALGORITHM

#include <opencv2/imgproc.hpp>

class WatershedSegmenter
{
private:
    cv::Mat markers;

public:
    // 类的 get 和 set 方法
    void setMarkers(const cv::Mat &markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(const cv::Mat &image)
    {
        cv::watershed(image, markers);
        return markers;
    }

    cv::Mat getSegmentation() const
    {
        cv::Mat tmp;
        // all segment with label higher than 255
        // will be assigned value 255
        markers.convertTo(tmp, CV_8U);

        return tmp;
    }

    // Return watershed in the form of an image
    cv::Mat getWatersheds()
    {
        cv::Mat tmp;
        markers.convertTo(tmp, CV_8U, 255, 255);

        return tmp;
    }
};

#endif // WATERSHED_ALGORITHM