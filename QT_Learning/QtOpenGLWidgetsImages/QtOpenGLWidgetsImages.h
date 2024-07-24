#pragma once

#include <QtWidgets/QWidget>
#include "ui_QtOpenGLWidgetsImages.h"
#include "gl_image.h"

QT_BEGIN_NAMESPACE
namespace Ui { class QtOpenGLWidgetsImagesClass; };
QT_END_NAMESPACE

class QtOpenGLWidgetsImages : public QWidget
{
    Q_OBJECT

public:
    QtOpenGLWidgetsImages(QWidget* parent = nullptr);
    ~QtOpenGLWidgetsImages();

private slots:
    void slotTimeOut();

private:
    Ui::QtOpenGLWidgetsImagesClass* ui;
    GL_Image* glImage;
    QTimer timer;
};
