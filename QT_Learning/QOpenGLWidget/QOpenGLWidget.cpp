#include "QOpenGLWidget.h"

CQOpenGLWidget::CQOpenGLWidget(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QOpenGLWidgetClass())
{
    ui->setupUi(this);

    connect(&timer, &QTimer::timeout, this, &CQOpenGLWidget::slotUpdate);
    timer.start(40);
}

CQOpenGLWidget::~CQOpenGLWidget()
{
    delete ui;
}

void CQOpenGLWidget::slotUpdate()
{
    ui->widget->setImage(QImage(R"(D:\Development\IntelligenceVisionBack\testImages/1EA_1201173129_324_6.jpeg)"));
}
