#pragma once

#include "ui_BarChart.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkChartXY.h>
#include <vtkContextView.h>
#include <vtkGenericOpenGLRenderWindow.h>

#include <QWidget>

/*
 * See "The Single Inheritance Approach" in this link:
 * [Using a Designer UI File in Your C++
 * Application](https://doc.qt.io/qt-5/designer-using-a-ui-file.html)
 */
QT_BEGIN_NAMESPACE

namespace Ui {
class BarChartForm;
}

QT_END_NAMESPACE

namespace Ithaca {

class BarChart : public QWidget
{
    Q_OBJECT

public:
    void Run();

public:
    // Constructor/Destructor
    explicit BarChart(QWidget *parent = nullptr);
    virtual ~BarChart();

private:
    // Designer form
    Ui::BarChartForm *ui = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow = nullptr;
    vtkSmartPointer<vtkChartXY>                   mpChart        = nullptr;
    vtkSmartPointer<vtkContextView>               mpView         = nullptr;
};

} // namespace Ithaca