#include "utility.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace WeiLi::Utility {

namespace {
inline QImage mat_to_qimage_ref(cv::Mat &mat, QImage::Format format)
{
    return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), format);
}

inline cv::Mat qimage_to_mat_ref(QImage &img, int format)
{
    return cv::Mat(img.height(), img.width(), format, img.bits(), img.bytesPerLine());
}
} //end of namespace

/**
 *@brief make Qimage and cv::Mat share the same buffer, the resource
 * of the cv::Mat must not deleted before the QImage finish
 * the jobs.
 *
 *@param mat : input mat
 *@param swap : true : swap BGR to RGB; false, do nothing
 */
QImage mat_to_qimage_ref(cv::Mat &mat, bool swap)
{
    if (!mat.empty())
    {
        switch (mat.type())
        {
        case CV_8UC3:
        {
            if (swap)
            {
                return mat_to_qimage_ref(mat, QImage::Format_RGB888).rgbSwapped();
            }
            else
            {
                return mat_to_qimage_ref(mat, QImage::Format_RGB888);
            }
        }

        case CV_8U:
        {
            return mat_to_qimage_ref(mat, QImage::Format_Indexed8);
        }

        case CV_8UC4:
        {
            return mat_to_qimage_ref(mat, QImage::Format_ARGB32);
        }
        }
    }

    return {};
}

/**
 *@brief copy cv::Mat into QImage
 *
 *@param mat : input mat
 *@param swap : true : swap BGR to RGB; false, do nothing
 */
QImage mat_to_qimage_cpy(const cv::Mat &mat, bool swap)
{
    return mat_to_qimage_ref(const_cast<cv::Mat &>(mat), swap).copy();
}

/**
 *@brief transform QImage to cv::Mat by sharing the buffer
 *@param img : input image
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
cv::Mat qimage_to_mat_ref(QImage &img, bool swap)
{
    if (img.isNull())
    {
        return cv::Mat();
    }

    switch (img.format())
    {
    case QImage::Format_RGB888:
    {
        auto result = qimage_to_mat_ref(img, CV_8UC3);
        if (swap)
        {
            cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
        }
        return result;
    }
    case QImage::Format_Indexed8:
    {
        return qimage_to_mat_ref(img, CV_8U);
    }
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
    {
        return qimage_to_mat_ref(img, CV_8UC4);
    }
    default:
        break;
    }

    return {};
}

/**
 *@brief transform QImage to cv::Mat by copy QImage to cv::Mat
 *@param img : input image
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
cv::Mat qimage_to_mat_cpy(const QImage &img, bool swap)
{
    return qimage_to_mat_ref(const_cast<QImage &>(img), swap).clone();
}

} // namespace WeiLi::Utility