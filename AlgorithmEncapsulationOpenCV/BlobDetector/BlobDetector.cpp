#include "BlobDetector.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>

#ifdef DEBUG_BLOB_DETECTOR
#    include "opencv2/highgui.hpp"
#endif

namespace SmartUltra { namespace BlobDetectorModule {

void BlobDetect::Run()
{
    if (m_inputImage.empty())
    {
        //        throw "Input image is empty.";
        CV_Error(cv::Error::StsNullPtr, "Input image is empty!");
        //        return;
    }
    this->Detect(m_inputImage);
}

void BlobDetect::SetParams(std::string name, float value)
{
    if ("minThreshold" == name)
    {
        m_params.minThreshold = value;
    }
    else if ("maxThreshold" == name)
    {
        m_params.maxThreshold = value;
    }
    else if ("thresholdStep" == name)
    {
        m_params.thresholdStep = value;
    }
    else if ("minRepeatability" == name)
    {
        m_params.minRepeatability = static_cast<size_t>(value);
    }
    else if ("minDistBetweenBlobs" == name)
    {
        m_params.minDistBetweenBlobs = value;
    }
    else if ("filterByColor" == name)
    {
        m_params.filterByColor = static_cast<bool>(value);
    }
    else if ("filterByArea" == name)
    {
        m_params.filterByArea = static_cast<bool>(value);
    }
    else if ("minArea" == name)
    {
        m_params.minArea = value;
    }
    else if ("maxArea" == name)
    {
        m_params.maxArea = value;
    }
    else if ("filterByCircularity" == name)
    {
        m_params.filterByCircularity = value;
    }
    else if ("minCircularity" == name)
    {
        m_params.minCircularity = value;
    }
    else if ("maxCircularity" == name)
    {
        m_params.maxCircularity = value;
    }
    else if ("filterByInertia" == name)
    {
        m_params.filterByInertia = static_cast<bool>(value);
    }
    else if ("minInertiaRatio" == name)
    {
        m_params.minInertiaRatio = value;
    }
    else if ("maxInertiaRatio" == name)
    {
        m_params.maxInertiaRatio = value;
    }
    else if ("filterByConvexity" == name)
    {
        m_params.filterByConvexity = static_cast<bool>(value);
    }
    else if ("minConvexity" == name)
    {
        m_params.minConvexity = value;
    }
    else if ("maxConvexity" == name)
    {
        m_params.maxConvexity = value;
    }
}

cv::Ptr<BlobDetect> BlobDetect::CreateInstance(const cv::SimpleBlobDetector::Params &parameters)
{
    return cv::makePtr<BlobDetect>(parameters);
}

void BlobDetect::Detect(const cv::_InputArray &image, const cv::_InputArray &mask)
{
    m_keyPoints.clear();
    CV_Assert(m_params.minRepeatability != 0);
    cv::Mat grayscaleImage;
    if (image.channels() == 3 || image.channels() == 4)
        cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
    else
        grayscaleImage = image.getMat();

    if (grayscaleImage.type() != CV_8UC1)
    {
        CV_Error(cv::Error::StsUnsupportedFormat, "Blob detector only supports 8-bit images!");
    }

    std::vector<std::vector<BlobInfo>> centers;

    for (double thresh = m_params.minThreshold; thresh < m_params.maxThreshold; thresh += m_params.thresholdStep)
    {
        cv::Mat binarizedImage;
        cv::threshold(grayscaleImage, binarizedImage, thresh, 255, cv::THRESH_BINARY);

        std::vector<BlobInfo> curCenters;
        FindBlobs(grayscaleImage, binarizedImage, curCenters);
        std::vector<std::vector<BlobInfo>> newCenters;
        for (size_t i = 0; i < curCenters.size(); i++)
        {
            bool isNew = true;
            for (size_t j = 0; j < centers.size(); j++)
            {
                double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
                isNew       = dist >= m_params.minDistBetweenBlobs && dist >= centers[j][centers[j].size() / 2].radius
                     && dist >= curCenters[i].radius;
                if (!isNew)
                {
                    centers[j].push_back(curCenters[i]);

                    size_t k = centers[j].size() - 1;
                    while (k > 0 && curCenters[i].radius < centers[j][k - 1].radius)
                    {
                        centers[j][k] = centers[j][k - 1];
                        k--;
                    }
                    centers[j][k] = curCenters[i];

                    break;
                }
            }
            if (isNew)
                newCenters.push_back(std::vector<BlobInfo>(1, curCenters[i]));
        }
        std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
    }

    // parse centers to result
    m_resultVec.blobList.clear();
    for (size_t i = 0; i < centers.size(); i++)
    {
        if (centers[i].size() < m_params.minRepeatability)
            continue;
        cv::Point2d sumPoint(0, 0);
        double      normalizer = 0;
        double      sumArea    = 0;
        for (size_t j = 0; j < centers[i].size(); j++)
        {
            sumPoint += centers[i][j].confidence * centers[i][j].location;
            normalizer += centers[i][j].confidence;
            sumArea += centers[i][j].area;
        }
        sumPoint *= (1. / normalizer);
        sumArea *= (1. / centers[i].size());
        normalizer *= (1. / centers[i].size());
        cv::KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius) * 2.0f);
        // parse centers to result
        BlobInfo     bi = centers[i][centers[i].size() / 2];
        bi.location     = sumPoint;
        bi.area         = sumArea;
        bi.confidence   = normalizer;
        m_resultVec.blobList.push_back(bi);
        m_keyPoints.push_back(kpt);
    }

    if (!mask.empty())
    {
        cv::KeyPointsFilter::runByPixelsMask(m_keyPoints, mask.getMat());
    }
}

void BlobDetect::FindBlobs(const cv::_InputArray &_image, const cv::_InputArray &_binaryImage,
                           std::vector<BlobInfo> &centers) const
{
    cv::Mat image = _image.getMat(), binaryImage = _binaryImage.getMat();
    CV_UNUSED(image);
    centers.clear();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
    cv::Mat keypointsImage;
    cv::cvtColor(binaryImage, keypointsImage, COLOR_GRAY2RGB);

    cv::Mat contoursImage;
    cv::cvtColor(binaryImage, contoursImage, COLOR_GRAY2RGB);
    cv::drawContours(contoursImage, contours, -1, Scalar(0, 255, 0));
    cv::imshow("contours", contoursImage);
#endif

    for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {
        BlobInfo center;
        center.confidence = 1;

        cv::Moments moms = cv::moments(contours[contourIdx]);
        if (m_params.filterByArea)
        {
            double area = moms.m00;
            if (area < m_params.minArea || area >= m_params.maxArea)
                continue;
        }

        if (m_params.filterByCircularity)
        {
            double area      = moms.m00;
            double perimeter = cv::arcLength(contours[contourIdx], true);
            double ratio     = 4 * CV_PI * area / (perimeter * perimeter);
            if (ratio < m_params.minCircularity || ratio >= m_params.maxCircularity)
                continue;
        }

        if (m_params.filterByInertia)
        {
            double       denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
            const double eps         = 1e-2;
            double       ratio;
            if (denominator > eps)
            {
                double cosmin = (moms.mu20 - moms.mu02) / denominator;
                double sinmin = 2 * moms.mu11 / denominator;
                double cosmax = -cosmin;
                double sinmax = -sinmin;

                double imin
                    = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
                double imax
                    = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
                ratio = imin / imax;
            }
            else
            {
                ratio = 1;
            }

            if (ratio < m_params.minInertiaRatio || ratio >= m_params.maxInertiaRatio)
                continue;

            center.confidence = ratio * ratio;
        }

        if (m_params.filterByConvexity)
        {
            std::vector<cv::Point> hull;
            cv::convexHull(contours[contourIdx], hull);
            double area     = cv::contourArea(contours[contourIdx]);
            double hullArea = cv::contourArea(hull);
            if (fabs(hullArea) < DBL_EPSILON)
                continue;
            double ratio = area / hullArea;
            if (ratio < m_params.minConvexity || ratio >= m_params.maxConvexity)
                continue;
        }

        if (moms.m00 == 0.0)
            continue;
        center.location = cv::Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

        if (m_params.filterByColor)
        {
            if (binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != m_params.blobColor)
                continue;
        }
        // area
        center.area = moms.m00;
        //compute blob radius
        {
            std::vector<double> dists;
            for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
            {
                cv::Point2d pt = contours[contourIdx][pointIdx];
                dists.push_back(norm(center.location - pt));
            }
            std::sort(dists.begin(), dists.end());
            center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        }
        //get blob outline
        center.outline.clear();
        // save blob outline
        center.outline.assign(contours[contourIdx].begin(), contours[contourIdx].end());
        centers.push_back(center);
#ifdef DEBUG_BLOB_DETECTOR
        cv::circle(keypointsImage, center.location, 1, cv::Scalar(0, 0, 255), 1);
#endif
    }
#ifdef DEBUG_BLOB_DETECTOR
    cv::imshow("bk", keypointsImage);
    cv::waitKey();
#endif
}

BlobDetect::BlobDetect(const cv::SimpleBlobDetector::Params &parameters)
    : m_params(parameters)
{
}

void BlobDetect::setParams(const SimpleBlobDetector::Params &params)
{
    m_params = params;
}

cv::SimpleBlobDetector::Params BlobDetect::getParams() const
{
    return m_params;
}

// Set parameters as default values
void BlobDetect::SetParams()
{
    //    -----------default values------------
    //    thresholdStep = 10;
    //    minThreshold = 50;
    //    maxThreshold = 220;
    //    minRepeatability = 2;
    //    minDistBetweenBlobs = 10;
    //
    //    filterByColor = true;
    //    blobColor = 0;
    //
    //    filterByArea = true;
    //    minArea = 25;
    //    maxArea = 5000;
    //
    //    filterByCircularity = false;
    //    minCircularity = 0.8f;
    //    maxCircularity = std::numeric_limits<float>::max();
    //
    //    filterByInertia = true;
    //    //minInertiaRatio = 0.6;
    //    minInertiaRatio = 0.1f;
    //    maxInertiaRatio = std::numeric_limits<float>::max();
    //
    //    filterByConvexity = true;
    //    //minConvexity = 0.8;
    //    minConvexity = 0.95f;
    //    maxConvexity = std::numeric_limits<float>::max();
    const cv::SimpleBlobDetector::Params &parameters = cv::SimpleBlobDetector::Params();
    m_params                                         = parameters;
}

void BlobDetect::Init(cv::Mat &_inputImage)
{
    m_inputImage = _inputImage;
}

void BlobDetect::PrintResultInfo() const
{
    if (m_resultVec.blobList.empty())
        std::cout << "The result is empty." << std::endl;

    std::cout << "Blob num: " << m_resultVec.blobList.size() << std::endl;
    int i = 0;
    for (const auto &r : m_resultVec.blobList)
    {
        ++i;
        std::cout << i << ": location:(" << r.location.x << "," << r.location.y << ") radius: " << r.radius
                  << " area: " << r.area << std::endl;
    }
}

void BlobDetect::PrintParameter() const
{
    std::cout << "Parameter:" << std::endl;
    std::cout << "thresholdStep:" << m_params.thresholdStep << std::endl;
    std::cout << "minThreshold:" << m_params.minThreshold << std::endl;
    std::cout << "maxThreshold:" << m_params.maxThreshold << std::endl;

    std::cout << "minDistBetweenBlobs:" << m_params.minDistBetweenBlobs << std::endl;
    std::cout << "minRepeatability:" << m_params.minRepeatability << std::endl;

    std::cout << "filterByConvexity:" << m_params.filterByConvexity << std::endl;
    std::cout << "maxConvexity:" << m_params.maxConvexity << std::endl;
    std::cout << "minConvexity:" << m_params.minConvexity << std::endl;

    std::cout << "filterByInertia:" << m_params.filterByInertia << std::endl;
    std::cout << "maxInertiaRatio:" << m_params.maxInertiaRatio << std::endl;
    std::cout << "minInertiaRatio:" << m_params.minInertiaRatio << std::endl;

    std::cout << "filterByCircularity:" << m_params.filterByCircularity << std::endl;
    std::cout << "maxCircularity:" << m_params.maxCircularity << std::endl;
    std::cout << "minCircularity:" << m_params.minCircularity << std::endl;

    std::cout << "filterByArea:" << m_params.filterByArea << std::endl;
    std::cout << "maxArea:" << m_params.maxArea << std::endl;
    std::cout << "minArea:" << m_params.minArea << std::endl;

    std::cout << "filterByColor:" << m_params.filterByColor << std::endl;
    std::cout << "blobColor:" << m_params.blobColor << std::endl;
}

void BlobDetect::DrawOutline() const
{
    cv::Mat img = m_inputImage.clone();
    if (img.channels() == 1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < m_resultVec.blobList.size(); ++i)
    {
        for (int j = 0; j < m_resultVec.blobList[i].outline.size(); ++j)
        {
            int x = static_cast<int>(m_resultVec.blobList[i].outline[j].x);
            int y = static_cast<int>(m_resultVec.blobList[i].outline[j].y);

            img.at<cv::Vec3b>(y, x)[0] = 0;
            img.at<cv::Vec3b>(y, x)[1] = 255;
            img.at<cv::Vec3b>(y, x)[2] = 0;
        }
    }
    cv::imshow("outline", img);
    cv::waitKey(0);
}

}} // namespace SmartUltra::BlobDetectorModule
