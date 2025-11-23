#pragma once

#include "ui_ShareCamera.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkSmartPointer.h>

#include <QWidget>

/*
 * See "The Single Inheritance Approach" in this link:
 * [Using a Designer UI File in Your C++
 * Application](https://doc.qt.io/qt-5/designer-using-a-ui-file.html)
 */
QT_BEGIN_NAMESPACE

namespace Ui {
class ShareCameraForm;
}

QT_END_NAMESPACE

namespace Ithaca {

class ShareCamera : public QWidget
{
    Q_OBJECT

public:
    void Run();

    void SetMessageCallback(std::function<void(bool, const std::string &)> callback);

public:
    // Constructor/Destructor
    explicit ShareCamera(QWidget *parent = nullptr);
    virtual ~ShareCamera();

protected:
    void ModifiedHandler();

private:
    // Designer form
    Ui::ShareCameraForm *ui = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidgetLeft  = nullptr;
    QVTKOpenGLNativeWidget *mpVtkRenderWidgetRight = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindowLeft  = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindowRight = nullptr;

    std::function<void(bool, const std::string &)> mCallbackFunc;
};

} // namespace Ithaca