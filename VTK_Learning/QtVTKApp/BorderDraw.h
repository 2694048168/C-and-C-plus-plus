#pragma once

#include "ui_BorderDraw.h"

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
class BorderDrawForm;
}

QT_END_NAMESPACE

namespace Ithaca {

class BorderDraw : public QWidget
{
    Q_OBJECT

public:
    void Run();

public:
    // Constructor/Destructor
    explicit BorderDraw(QWidget *parent = nullptr);
    virtual ~BorderDraw();

private:
    // Designer form
    Ui::BorderDrawForm *ui = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow = nullptr;
    vtkSmartPointer<vtkChartXY>                   mpChart        = nullptr;
    vtkSmartPointer<vtkContextView>               mpView         = nullptr;
};

} // namespace Ithaca