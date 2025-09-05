#include "ImageOperator/ImageEdgeDetector.h"

namespace Ithaca {

bool ImageEdgeDetector::Run(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC1);
    if (srcImg.type() != CV_8UC1)
        return false;

    if (srcImg.empty())
        return false;

    cv::Mat sobelx, sobely, gradient;

    // Apply Sobel operator
    cv::Sobel(srcImg, sobelx, CV_64F, 1, 0, 3);
    cv::Sobel(srcImg, sobely, CV_64F, 0, 1, 3);

    // Compute gradient magnitude
    cv::magnitude(sobelx, sobely, gradient);

    // Convert to 8-bit image
    cv::convertScaleAbs(gradient, dstImg);

    return true;
}

bool ImageEdgeDetector::RunExt(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC1);
    if (srcImg.type() != CV_8UC1)
        return false;

    if (srcImg.empty())
        return false;

    dstImg.create(srcImg.size(), srcImg.type());
    for (int y = 1; y < srcImg.rows - 1; y++)
    {
        const uchar *src_row = srcImg.ptr<uchar>(y);
        uchar       *dst_row = dstImg.ptr<uchar>(y);
        for (int x = 1; x < srcImg.cols - 1; x++)
        {
            // 3×3 Sobel-like卷积核
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++)
            {
                const uchar *src_prev_row = srcImg.ptr<uchar>(y + ky);
                // ?BRG-three channels ----> *3 pointer address offset
                // sum += src_prev_row[(x - 1) * 3] * (-1) + src_prev_row[x * 3] * 0 + src_prev_row[(x + 1) * 3] * 1;

                sum += src_prev_row[x - 1] * (-1) + src_prev_row[x] * 0 + src_prev_row[x + 1] * 1;
            }
            dst_row[x] = cv::saturate_cast<uchar>(abs(sum));
        }
    }

    return true;
}

bool ImageEdgeDetector::RunExtOpti(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC1);
    if (srcImg.type() != CV_8UC1)
        return false;

    if (srcImg.empty())
        return false;

    dstImg.create(srcImg.size(), srcImg.type());
    int cols = srcImg.cols;
// 多线程处理图像行
#pragma omp parallel for num_threads(8)
    for (int y = 1; y < srcImg.rows - 1; y++)
    {
        uchar *dst_row = dstImg.ptr<uchar>(y);
        // AVX2指令集优化内层循环
        for (int x = 1; x < cols - 1; x += 8)
        {
            __m256i sum = _mm256_setzero_si256();
            // 向量化计算8个像素的梯度
            for (int ky = -1; ky <= 1; ky++)
            {
                // ?BRG-three channels ----> *3 pointer address offset
                // __m256i left = _mm256_loadu_si256((__m256i*)&src.ptr<uchar>(y+ky)[(x-1)*3]);
                // __m256i right = _mm256_loadu_si256((__m256i*)&src.ptr<uchar>(y+ky)[(x+1)*3]);

                __m256i left  = _mm256_loadu_si256((__m256i *)&srcImg.ptr<uchar>(y + ky)[(x - 1) * 3]);
                __m256i right = _mm256_loadu_si256((__m256i *)&srcImg.ptr<uchar>(y + ky)[(x + 1) * 3]);
                sum           = _mm256_add_epi8(sum, _mm256_sub_epi8(right, left));
            }
            // 结果存储
            _mm256_storeu_si256((__m256i *)&dst_row[x], _mm256_abs_epi8(sum));
        }
    }

    return true;
}

bool image_edge_detector(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC1);
    if (srcImg.type() != CV_8UC1)
        return false;

    if (srcImg.empty())
        return false;

    cv::Mat sobelx, sobely, gradient;

    // Apply Sobel operator
    cv::Sobel(srcImg, sobelx, CV_64F, 1, 0, 3);
    cv::Sobel(srcImg, sobely, CV_64F, 0, 1, 3);

    // Compute gradient magnitude
    cv::magnitude(sobelx, sobely, gradient);

    // Convert to 8-bit image
    cv::convertScaleAbs(gradient, dstImg);

    return true;
}

bool image_edge_detector_Ext(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC1);
    if (srcImg.type() != CV_8UC1)
        return false;

    if (srcImg.empty())
        return false;

    dstImg.create(srcImg.size(), srcImg.type());
    for (int y = 1; y < srcImg.rows - 1; y++)
    {
        const uchar *src_row = srcImg.ptr<uchar>(y);
        uchar       *dst_row = dstImg.ptr<uchar>(y);
        for (int x = 1; x < srcImg.cols - 1; x++)
        {
            // 3×3 Sobel-like卷积核
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++)
            {
                const uchar *src_prev_row = srcImg.ptr<uchar>(y + ky);
                // ?BRG-three channels ----> *3 pointer address offset
                // sum += src_prev_row[(x - 1) * 3] * (-1) + src_prev_row[x * 3] * 0 + src_prev_row[(x + 1) * 3] * 1;

                sum += src_prev_row[x - 1] * (-1) + src_prev_row[x] * 0 + src_prev_row[x + 1] * 1;
            }
            dst_row[x] = cv::saturate_cast<uchar>(abs(sum));
        }
    }

    return true;
}

bool image_edge_detector_ExtOpti(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC1);
    if (srcImg.type() != CV_8UC1)
        return false;

    if (srcImg.empty())
        return false;

    dstImg.create(srcImg.size(), srcImg.type());
    int cols = srcImg.cols;
// 多线程处理图像行
#pragma omp parallel for num_threads(8)
    for (int y = 1; y < srcImg.rows - 1; y++)
    {
        uchar *dst_row = dstImg.ptr<uchar>(y);
        // AVX2指令集优化内层循环
        for (int x = 1; x < cols - 1; x += 8)
        {
            __m256i sum = _mm256_setzero_si256();
            // 向量化计算8个像素的梯度
            for (int ky = -1; ky <= 1; ky++)
            {
                // ?BRG-three channels ----> *3 pointer address offset
                // __m256i left = _mm256_loadu_si256((__m256i*)&src.ptr<uchar>(y+ky)[(x-1)*3]);
                // __m256i right = _mm256_loadu_si256((__m256i*)&src.ptr<uchar>(y+ky)[(x+1)*3]);

                __m256i left  = _mm256_loadu_si256((__m256i *)&srcImg.ptr<uchar>(y + ky)[x - 1]);
                __m256i right = _mm256_loadu_si256((__m256i *)&srcImg.ptr<uchar>(y + ky)[x + 1]);
                sum           = _mm256_add_epi8(sum, _mm256_sub_epi8(right, left));
            }
            // 结果存储
            _mm256_storeu_si256((__m256i *)&dst_row[x], _mm256_abs_epi8(sum));
        }
    }

    return true;
}

} // namespace Ithaca
