#pragma once

#include <QtWidgets/QWidget>
#include "ui_QOpenGLWidget.h"
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui { class QOpenGLWidgetClass; };
QT_END_NAMESPACE

class CQOpenGLWidget : public QWidget
{
    Q_OBJECT

public:
    CQOpenGLWidget(QWidget* parent = nullptr);
    ~CQOpenGLWidget();

public slots:
    void slotUpdate();

private:
    Ui::QOpenGLWidgetClass* ui;
    QTimer timer;

};
