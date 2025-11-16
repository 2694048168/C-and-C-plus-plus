#pragma once

#include "ui_DeformSphere.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkDataSetMapper.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkSphereSource.h>

#include <QWidget>
#include <cmath>
#include <random>

QT_BEGIN_NAMESPACE

namespace Ui {
class DeformSphereForm;
};

QT_END_NAMESPACE

namespace Ithaca {

class DeformSphere : public QWidget
{
    Q_OBJECT

public:
    void Run();

    /**
     * Deform the sphere source using a random amplitude and modes and render it in
     * the window
     *
     * @param sphere the original sphere source
     * @param mapper the mapper for the scene
     * @param window the window to render to
     * @param randEng the random number generator engine
     */
    void Randomize();

public:
    DeformSphere(QWidget *parent = nullptr);
    ~DeformSphere();

private:
    Ui::DeformSphereForm *ui;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpWindow = nullptr;
    vtkSmartPointer<vtkSphereSource>              mpSphere = nullptr;
    vtkSmartPointer<vtkDataSetMapper>             mpMapper = nullptr;

    std::mt19937 mRandEng;
};

} // namespace Ithaca
