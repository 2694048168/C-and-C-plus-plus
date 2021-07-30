/**
 * @File    : imageComparator.hpp
 * @Brief   : 比较直方图搜索相似图像 基于内容的图像检索是计算机视觉的一个重要课题
 *      它包括根据一个已有的基准图像，找出一批内容相似的图像
 *      直方图是标识图像内容的一种有效方式，因此值得研究一下能否用它来解决基于内容的图像检索问题。
 *      关键是，要仅靠比较它们的直方图就测量出两幅图像的相似度
 *      需要定义一个测量函数，来评估两个直方图之间的差异程度或相似程度
 * OpenCV 在 cv::compareHist 函数的实现过程中使用了其中的一些方法: 
 *  1. 卡方测量法（ cv::HISTCMP_CHISQR 标志）累加各箱子的归一化平方差；
 *  2. 关联性算法（ cv::HISTCMP_CORREL 标志）基于信号处理中的归一化交叉关联操作符测量两个信号的相似度； 
 *  3. Bhattacharyya 测量法（ cv::HISTCMP_BHATTACHARYYA 标志）和 
 *  4. Kullback-Leibler发散度（ cv::HISTCMP_KL_DIV 标志）都用在统计学中，评估两个概率分布的相似度。
 * 
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#ifndef IMG_COMPARATOR
#define IMG_COMPARATOR

#include <opencv2/imgproc/imgproc.hpp>
#include "colorhistogram.hpp"

class ImageCompatator
{
private:
    cv::Mat refH;        // 基准直方图
    cv::Mat inputH;      // 输入图像的直方图
    ColorHistogram hist; // 生成直方图
    int num_bins;        // 每个颜色通道使用的箱子数量 为了得到更加可靠的相似度测量结果，需要在计算直方图时减少箱子的数量

public:
    // 无参默认构造函数
    ImageCompatator(): num_bins(8){}

    // 类的 get 和 set 方法
    void setNumberOfBins(int bins)
    {
        num_bins = bins;
    }
    int getNumberOfBins() const
    {
        return num_bins;
    }

    void setReferenceImage(const cv::Mat &image)
    {
        hist.setSize(num_bins);
        refH = hist.getHistogram(image);
    }

    // compare the image using their BGR histograms
    double compare(const cv::Mat &image)
    {
        inputH = hist.getHistogram(image);
        // histogram comparison using intersection 使用交集的直方图比较
        return cv::compareHist(refH, inputH, cv::HISTCMP_INTERSECT);
    }
};

#endif // IMG_COMPARATOR
