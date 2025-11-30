#pragma once

#include "ui_MedicalModel.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkMetaImageReader.h>
#include <vtkObject.h>
#include <vtkSmartPointer.h>

#include <QWidget>
#include <string>

/*
 * See "The Single Inheritance Approach" in this link:
 * [Using a Designer UI File in Your C++
 * Application](https://doc.qt.io/qt-5/designer-using-a-ui-file.html)
 */
QT_BEGIN_NAMESPACE

namespace Ui {
class MedicalModelForm;
}

QT_END_NAMESPACE

namespace Ithaca {

class MedicalModel : public QWidget
{
    Q_OBJECT

public:
    void Run();

    void SetMessageCallback(std::function<void(bool, const std::string &)> callback);

public:
    // Constructor/Destructor
    explicit MedicalModel(QWidget *parent = nullptr);
    virtual ~MedicalModel();

public slots:
    void sl_LoadModelFilepath();

private:
    // Designer form
    Ui::MedicalModelForm *ui = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow = nullptr;

    vtkSmartPointer<vtkMetaImageReader> mpReader = nullptr;

    std::function<void(bool, const std::string &)> mCallbackFunc = nullptr;

    std::string mFilepath = "";
};

} // namespace Ithaca