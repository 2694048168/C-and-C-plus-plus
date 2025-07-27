#include "BlobToolControl.h"

BlobToolControl::BlobToolControl(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    connectSignals();
}

void BlobToolControl::setImage(const cv::Mat &image)
{
    cv::cvtColor(image, this->image, cv::COLOR_BGR2RGB);
    displayImage();
}

void BlobToolControl::onLoadImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.jpg *.bmp)");
    if (!fileName.isEmpty())
    {
        cv::Mat image = cv::imread(fileName.toStdString());
        setImage(image);
    }
}

void BlobToolControl::onDetect()
{
    if (image.empty())
        return;

    BlobParams params;
    params.minThreshold        = minThresholdSlider->value();
    params.maxThreshold        = maxThresholdSlider->value();
    params.minArea             = minAreaSlider->value();
    params.maxArea             = maxAreaSlider->value();
    params.filterByColor       = filterByColorCheckBox->isChecked();
    params.blobColor           = blobColorSpinBox->value();
    params.filterByCircularity = filterByCircularityCheckBox->isChecked();
    params.minCircularity      = minCircularitySlider->value() / 100.0f;
    params.filterByConvexity   = filterByConvexityCheckBox->isChecked();
    params.minConvexity        = minConvexitySlider->value() / 100.0f;
    params.filterByInertia     = filterByInertiaCheckBox->isChecked();
    params.minInertiaRatio     = minInertiaRatioSlider->value() / 100.0f;

    analyzer.setParams(params);
    auto keypoints = analyzer.detect(image);

    cv::Mat resultImage;
    cv::drawKeypoints(image, keypoints, resultImage, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    setImage(resultImage);
}

void BlobToolControl::setupUI()
{
    QVBoxLayout *layout = new QVBoxLayout(this);

    // 添加控件
    minThresholdSlider          = new QSlider(Qt::Horizontal);
    maxThresholdSlider          = new QSlider(Qt::Horizontal);
    minAreaSlider               = new QSlider(Qt::Horizontal);
    maxAreaSlider               = new QSlider(Qt::Horizontal);
    filterByColorCheckBox       = new QCheckBox("Filter by Color");
    blobColorSpinBox            = new QSpinBox();
    filterByCircularityCheckBox = new QCheckBox("Filter by Circularity");
    minCircularitySlider        = new QSlider(Qt::Horizontal);
    filterByConvexityCheckBox   = new QCheckBox("Filter by Convexity");
    minConvexitySlider          = new QSlider(Qt::Horizontal);
    filterByInertiaCheckBox     = new QCheckBox("Filter by Inertia");
    minInertiaRatioSlider       = new QSlider(Qt::Horizontal);

    QPushButton *loadImageButton = new QPushButton("Load Image");
    QPushButton *detectButton    = new QPushButton("Detect");

    layout->addWidget(new QLabel("Min Threshold"));
    layout->addWidget(minThresholdSlider);
    layout->addWidget(new QLabel("Max Threshold"));
    layout->addWidget(maxThresholdSlider);
    layout->addWidget(new QLabel("Min Area"));
    layout->addWidget(minAreaSlider);
    layout->addWidget(new QLabel("Max Area"));
    layout->addWidget(maxAreaSlider);
    layout->addWidget(filterByColorCheckBox);
    layout->addWidget(new QLabel("Blob Color"));
    layout->addWidget(blobColorSpinBox);
    layout->addWidget(filterByCircularityCheckBox);
    layout->addWidget(new QLabel("Min Circularity"));
    layout->addWidget(minCircularitySlider);
    layout->addWidget(filterByConvexityCheckBox);
    layout->addWidget(new QLabel("Min Convexity"));
    layout->addWidget(minConvexitySlider);
    layout->addWidget(filterByInertiaCheckBox);
    layout->addWidget(new QLabel("Min Inertia Ratio"));
    layout->addWidget(minInertiaRatioSlider);
    layout->addWidget(loadImageButton);
    layout->addWidget(detectButton);

    setLayout(layout);
}

void BlobToolControl::connectSignals()
{
    connect(loadImageButton, &QPushButton::clicked, this, &BlobToolControl::onLoadImage);
    connect(detectButton, &QPushButton::clicked, this, &BlobToolControl::onDetect);
}

void BlobToolControl::displayImage()
{
    QImage qImage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    imageLabel->setPixmap(QPixmap::fromImage(qImage));
}
