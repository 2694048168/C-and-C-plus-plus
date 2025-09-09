#include "ImageOperator/ImageTemplateMatch.h"

#include <immintrin.h> // AVX2指令头文件
#include <omp.h>       // 添加OpenMP支持

namespace Ithaca {

bool ImageTemplateMatch::Run(const cv::Mat &srcImg, const cv::Mat &templImg, cv::Mat &resultImg)
{
    if (srcImg.empty() || templImg.empty())
        return false;

    // 2. 创建用于存放结果的矩阵
    cv::Mat resultImage;
    int     result_cols = srcImg.cols - templImg.cols + 1;
    int     result_rows = srcImg.rows - templImg.rows + 1;
    resultImage.create(result_rows, result_cols, CV_32FC1);

    // 3. 执行模板匹配 (使用归一化相关系数法)
    cv::matchTemplate(srcImg, templImg, resultImage, cv::TM_CCOEFF_NORMED);

    // 4. 找到最佳匹配位置
    double    minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(resultImage, &minVal, &maxVal, &minLoc, &maxLoc);

    // 对于 TM_CCOEFF_NORMED 方法，最佳匹配点是最大值所在的位置
    cv::Point matchLoc = maxLoc;

    // 5. 在源图像上绘制矩形框来标记匹配区域
    resultImg = srcImg.clone();
    cv::rectangle(resultImg, matchLoc, cv::Point(matchLoc.x + templImg.cols, matchLoc.y + templImg.rows),
                  cv::Scalar(0, 255, 0), 7); // BGR

    return true;
}

bool ImageTemplateMatch::RunOpti(const cv::Mat &srcImg, const cv::Mat &templImg, cv::Mat &resultImg)
{
    if (srcImg.empty() || templImg.empty())
        return false;

    // 确保图像是浮点类型（AVX指令需要）
    cv::Mat srcFloat, templFloat;
    if (srcImg.type() != CV_32F)
        srcImg.convertTo(srcFloat, CV_32F);
    else
        srcFloat = srcImg;

    if (templImg.type() != CV_32F)
        templImg.convertTo(templFloat, CV_32F);
    else
        templFloat = templImg;

    int img_h = srcFloat.rows;
    int img_w = srcFloat.cols;
    int t_h   = templFloat.rows;
    int t_w   = templFloat.cols;

    cv::Mat result;
    result.create(img_h - t_h + 1, img_w - t_w + 1, CV_32FC1);

    // 计算模板的均值和标准差（用于NCC）
    cv::Scalar t_mean, t_stddev;
    cv::meanStdDev(templFloat, t_mean, t_stddev);
    float templ_mean = static_cast<float>(t_mean[0]);
    float templ_std  = static_cast<float>(t_stddev[0]);

    // 设置AVX向量
    __m256 v_t_mean = _mm256_set1_ps(templ_mean);
    __m256 v_t_std  = _mm256_set1_ps(templ_std);

#pragma omp parallel for collapse(2) // 二维并行
    for (int y = 0; y < result.rows; y++)
    {
        float *res_row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; x += 8) // 每次处理8个像素（AVX256的宽度）
        {
            // 确保不越界
            if (x + 7 >= result.cols)
            {
                for (int xx = x; xx < result.cols; xx++)
                {
                    // 对于边缘部分，使用普通方法计算
                    cv::Mat    imgPatch = srcFloat(cv::Rect(xx, y, t_w, t_h)).clone();
                    cv::Scalar i_mean, i_stddev;
                    cv::meanStdDev(imgPatch, i_mean, i_stddev);

                    float numerator = 0.0f;
                    for (int i = 0; i < t_h; i++)
                    {
                        for (int j = 0; j < t_w; j++)
                        {
                            numerator += (srcFloat.at<float>(y + i, xx + j) - i_mean[0])
                                       * (templFloat.at<float>(i, j) - templ_mean);
                        }
                    }

                    float denominator = i_stddev[0] * templ_std;
                    res_row[xx]       = (denominator != 0) ? numerator / denominator : 0;
                }
                break;
            }

            // 初始化累加器
            __m256 v_numerator  = _mm256_setzero_ps();
            __m256 v_img_sum    = _mm256_setzero_ps();
            __m256 v_img_sq_sum = _mm256_setzero_ps();

            // 计算图像块的均值和标准差
            for (int i = 0; i < t_h; i++)
            {
                for (int j = 0; j < t_w; j++)
                {
                    // 加载8个图像块和模板值
                    __m256 v_img_val
                        = _mm256_setr_ps(srcFloat.at<float>(y + i, x + j), srcFloat.at<float>(y + i, x + 1 + j),
                                         srcFloat.at<float>(y + i, x + 2 + j), srcFloat.at<float>(y + i, x + 3 + j),
                                         srcFloat.at<float>(y + i, x + 4 + j), srcFloat.at<float>(y + i, x + 5 + j),
                                         srcFloat.at<float>(y + i, x + 6 + j), srcFloat.at<float>(y + i, x + 7 + j));

                    __m256 v_templ_val = _mm256_set1_ps(templFloat.at<float>(i, j));

                    // 累加图像块的和与平方和
                    v_img_sum    = _mm256_add_ps(v_img_sum, v_img_val);
                    v_img_sq_sum = _mm256_add_ps(v_img_sq_sum, _mm256_mul_ps(v_img_val, v_img_val));

                    // 计算分子部分
                    v_numerator = _mm256_add_ps(v_numerator, _mm256_mul_ps(_mm256_sub_ps(v_img_val, v_t_mean),
                                                                           _mm256_sub_ps(v_templ_val, v_t_mean)));
                }
            }

            // 计算图像块的均值
            __m256 v_i_mean = _mm256_div_ps(v_img_sum, _mm256_set1_ps(t_w * t_h));

            // 计算图像块的标准差
            __m256 v_i_std = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_div_ps(v_img_sq_sum, _mm256_set1_ps(t_w * t_h)),
                                                          _mm256_mul_ps(v_i_mean, v_i_mean)));

            // 计算分母
            __m256 v_denominator = _mm256_mul_ps(v_i_std, v_t_std);

            // 计算NCC
            __m256 v_ncc = _mm256_div_ps(v_numerator, v_denominator);

            // 存储结果
            _mm256_storeu_ps(&res_row[x], v_ncc);
        }
    }

    // 找到最佳匹配位置
    double    minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 对于归一化互相关系数，最佳匹配点是最大值所在的位置
    cv::Point matchLoc = maxLoc;

    // 在源图像上绘制矩形框来标记匹配区域
    resultImg = srcImg.clone();
    cv::rectangle(resultImg, matchLoc, cv::Point(matchLoc.x + templImg.cols, matchLoc.y + templImg.rows),
                  cv::Scalar(0, 255, 0), 7); // BGR

    return true;
}

bool image_template_match(const cv::Mat &srcImg, const cv::Mat &templImg, cv::Mat &resultImg)
{
    if (srcImg.empty() || templImg.empty())
        return false;

    // 2. 创建用于存放结果的矩阵
    cv::Mat resultImage;
    int     result_cols = srcImg.cols - templImg.cols + 1;
    int     result_rows = srcImg.rows - templImg.rows + 1;
    resultImage.create(result_rows, result_cols, CV_32FC1);

    // 3. 执行模板匹配 (使用归一化相关系数法)
    cv::matchTemplate(srcImg, templImg, resultImage, cv::TM_CCOEFF_NORMED);

    // 4. 找到最佳匹配位置
    double    minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(resultImage, &minVal, &maxVal, &minLoc, &maxLoc);

    // 对于 TM_CCOEFF_NORMED 方法，最佳匹配点是最大值所在的位置
    cv::Point matchLoc = maxLoc;

    // 5. 在源图像上绘制矩形框来标记匹配区域
    resultImg = srcImg.clone();
    cv::rectangle(resultImg, matchLoc, cv::Point(matchLoc.x + templImg.cols, matchLoc.y + templImg.rows),
                  cv::Scalar(0, 255, 0), 7); // BGR

    return true;
}

bool image_template_match_Opti(const cv::Mat &srcImg, const cv::Mat &templImg, cv::Mat &resultImg)
{
    if (srcImg.empty() || templImg.empty())
        return false;

    // 确保图像是浮点类型（AVX指令需要）
    cv::Mat srcFloat, templFloat;
    if (srcImg.type() != CV_32F)
        srcImg.convertTo(srcFloat, CV_32F);
    else
        srcFloat = srcImg;

    if (templImg.type() != CV_32F)
        templImg.convertTo(templFloat, CV_32F);
    else
        templFloat = templImg;

    int img_h = srcFloat.rows;
    int img_w = srcFloat.cols;
    int t_h   = templFloat.rows;
    int t_w   = templFloat.cols;

    cv::Mat result;
    result.create(img_h - t_h + 1, img_w - t_w + 1, CV_32FC1);

    // 计算模板的均值和标准差（用于NCC）
    cv::Scalar t_mean, t_stddev;
    cv::meanStdDev(templFloat, t_mean, t_stddev);
    float templ_mean = static_cast<float>(t_mean[0]);
    float templ_std  = static_cast<float>(t_stddev[0]);

    // 设置AVX向量
    __m256 v_t_mean = _mm256_set1_ps(templ_mean);
    __m256 v_t_std  = _mm256_set1_ps(templ_std);

#pragma omp parallel for collapse(2) // 二维并行
    for (int y = 0; y < result.rows; y++)
    {
        float *res_row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; x += 8) // 每次处理8个像素（AVX256的宽度）
        {
            // 确保不越界
            if (x + 7 >= result.cols)
            {
                for (int xx = x; xx < result.cols; xx++)
                {
                    // 对于边缘部分，使用普通方法计算
                    cv::Mat    imgPatch = srcFloat(cv::Rect(xx, y, t_w, t_h)).clone();
                    cv::Scalar i_mean, i_stddev;
                    cv::meanStdDev(imgPatch, i_mean, i_stddev);

                    float numerator = 0.0f;
                    for (int i = 0; i < t_h; i++)
                    {
                        for (int j = 0; j < t_w; j++)
                        {
                            numerator += (srcFloat.at<float>(y + i, xx + j) - i_mean[0])
                                       * (templFloat.at<float>(i, j) - templ_mean);
                        }
                    }

                    float denominator = i_stddev[0] * templ_std;
                    res_row[xx]       = (denominator != 0) ? numerator / denominator : 0;
                }
                break;
            }

            // 初始化累加器
            __m256 v_numerator  = _mm256_setzero_ps();
            __m256 v_img_sum    = _mm256_setzero_ps();
            __m256 v_img_sq_sum = _mm256_setzero_ps();

            // 计算图像块的均值和标准差
            for (int i = 0; i < t_h; i++)
            {
                for (int j = 0; j < t_w; j++)
                {
                    // 加载8个图像块和模板值
                    __m256 v_img_val
                        = _mm256_setr_ps(srcFloat.at<float>(y + i, x + j), srcFloat.at<float>(y + i, x + 1 + j),
                                         srcFloat.at<float>(y + i, x + 2 + j), srcFloat.at<float>(y + i, x + 3 + j),
                                         srcFloat.at<float>(y + i, x + 4 + j), srcFloat.at<float>(y + i, x + 5 + j),
                                         srcFloat.at<float>(y + i, x + 6 + j), srcFloat.at<float>(y + i, x + 7 + j));

                    __m256 v_templ_val = _mm256_set1_ps(templFloat.at<float>(i, j));

                    // 累加图像块的和与平方和
                    v_img_sum    = _mm256_add_ps(v_img_sum, v_img_val);
                    v_img_sq_sum = _mm256_add_ps(v_img_sq_sum, _mm256_mul_ps(v_img_val, v_img_val));

                    // 计算分子部分
                    v_numerator = _mm256_add_ps(v_numerator, _mm256_mul_ps(_mm256_sub_ps(v_img_val, v_t_mean),
                                                                           _mm256_sub_ps(v_templ_val, v_t_mean)));
                }
            }

            // 计算图像块的均值
            __m256 v_i_mean = _mm256_div_ps(v_img_sum, _mm256_set1_ps(t_w * t_h));

            // 计算图像块的标准差
            __m256 v_i_std = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_div_ps(v_img_sq_sum, _mm256_set1_ps(t_w * t_h)),
                                                          _mm256_mul_ps(v_i_mean, v_i_mean)));

            // 计算分母
            __m256 v_denominator = _mm256_mul_ps(v_i_std, v_t_std);

            // 计算NCC
            __m256 v_ncc = _mm256_div_ps(v_numerator, v_denominator);

            // 存储结果
            _mm256_storeu_ps(&res_row[x], v_ncc);
        }
    }

    // 找到最佳匹配位置
    double    minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 对于归一化互相关系数，最佳匹配点是最大值所在的位置
    cv::Point matchLoc = maxLoc;

    // 在源图像上绘制矩形框来标记匹配区域
    resultImg = srcImg.clone();
    cv::rectangle(resultImg, matchLoc, cv::Point(matchLoc.x + templImg.cols, matchLoc.y + templImg.rows),
                  cv::Scalar(0, 255, 0), 7); // BGR

    return true;
}

} // namespace Ithaca
