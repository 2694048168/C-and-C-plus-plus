#include "ImageRender.h"

#include "ui_ImageRender.h"

ImageRender::ImageRender(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::FImageRender)
    , m_pTimer{new QTimer}
{
    ui->setupUi(this);

    connect(m_pTimer, &QTimer::timeout, this, &ImageRender::sl_UpdateImage);
    m_pTimer->start(40);
}

ImageRender::~ImageRender()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

void ImageRender::sl_UpdateImage()
{
    ui->widget->setImage(QImage("D:/Development/IntelligenceVision/src/Images/CCD1/1EA_1201173129_324_6.jpeg"));
}
