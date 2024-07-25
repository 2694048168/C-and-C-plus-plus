#include "QtOpenGLWidgetsImages.h"
#include <iostream>

QtOpenGLWidgetsImages::QtOpenGLWidgetsImages(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QtOpenGLWidgetsImagesClass())
    , m_pImage{ nullptr }
    , m_idx{ 0 }
{
    ui->setupUi(this);

    // 将widget控件作为绘制窗口
    //glImage = new GL_Image(ui->widget);
    //glImage->setFixedSize(ui->widget->size());

    connect(&timer, SIGNAL(timeout()), this, SLOT(slotTimeOut()));
    timer.setTimerType(Qt::PreciseTimer);
    timer.start(100);
}

QtOpenGLWidgetsImages::~QtOpenGLWidgetsImages()
{
    delete ui;
}


// 主界面开启定时器，在界面循环显示4个方向的图片
void QtOpenGLWidgetsImages::slotTimeOut()
{
    // 需要修改为自己的图片路径
    auto imageName = R"(D:\Development\GitRepository\ComputerVisionDeepLearning\DigitalImageProcessing\image/lena.jpg)";
    m_pImage = new QImage(imageName);
    //m_pImage->rgbSwapped(); //qimage加载的颜色通道顺序和opengl显示的颜色通道顺序不一致,调换R通道和B通道
    *m_pImage = m_pImage->rgbSwapped(); //qimage加载的颜色通道顺序和opengl显示的颜色通道顺序不一致,调换R通道和B通道
    ui->widget->SetImage(m_pImage);

    ++m_idx;
    std::cout << "the " << m_idx << " image update\n";
}
