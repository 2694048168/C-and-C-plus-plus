#pragma once

#include <QtWidgets/QWidget>
#include "ui_QGLWidgetTexture.h"
#include "glwidget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class QGLWidgetTextureClass; };
QT_END_NAMESPACE

class QGLWidgetTexture : public QWidget
{
    Q_OBJECT

public:
    QGLWidgetTexture(QWidget* parent = nullptr);
    ~QGLWidgetTexture();

private slots:
    void setCurrentGlWidget();
    void rotateOneStep();

private:
    enum { NumRows = 2, NumColumns = 3 };

    GLWidget* glWidgets[NumRows][NumColumns];
    GLWidget* currentGlWidget;

private:
    Ui::QGLWidgetTextureClass* ui;
};
