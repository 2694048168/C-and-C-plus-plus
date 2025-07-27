#pragma once

#include "BlobTool.h"
#include "opencv2/opencv.hpp"

#include <QCheckBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>

class BlobToolControl : public QWidget
{
    Q_OBJECT

public:
    BlobToolControl(QWidget *parent = nullptr);

    void setImage(const cv::Mat &image);

private slots:

    void onLoadImage();

    void onDetect();

private:
    void setupUI();

    void connectSignals();

    void displayImage();

private:
    cv::Mat  image;
    BlobTool analyzer;

    QSlider     *minThresholdSlider;
    QSlider     *maxThresholdSlider;
    QSlider     *minAreaSlider;
    QSlider     *maxAreaSlider;
    QCheckBox   *filterByColorCheckBox;
    QSpinBox    *blobColorSpinBox;
    QCheckBox   *filterByCircularityCheckBox;
    QSlider     *minCircularitySlider;
    QCheckBox   *filterByConvexityCheckBox;
    QSlider     *minConvexitySlider;
    QCheckBox   *filterByInertiaCheckBox;
    QSlider     *minInertiaRatioSlider;
    QPushButton *loadImageButton;
    QPushButton *detectButton;
    QLabel      *imageLabel;
};
