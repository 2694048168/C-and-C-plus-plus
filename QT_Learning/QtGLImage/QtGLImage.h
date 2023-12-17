#pragma once

#include <QtWidgets/QWidget>
#include <QTime>
#include "ui_QtGLImage.h"
#include "GL_Image.hpp"
#include <string>
#include <vector>
#include "ThreadShowImage.h"

QT_BEGIN_NAMESPACE
namespace Ui { class QtGLImageClass; };
QT_END_NAMESPACE

class QtGLImage : public QWidget
{
    Q_OBJECT

public:
    QtGLImage(QWidget* parent = nullptr);
    ~QtGLImage();

private slots:
    void slotTimeOut();
    void sl_setImgPath();
    void sl_UpdateShowImage(QImage image);

private:
    Ui::QtGLImageClass* ui;
    std::string m_imgPath;
    std::vector<std::string> m_imgFilelist;
    GL_Image* glImage;
    QTimer timer;
    ThreadShowImage* m_pThreadShowImg;
};
