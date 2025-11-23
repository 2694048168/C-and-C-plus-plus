#include "ShareCamera.h"

#include <vtkAutoInit.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkConeSource.h>
#include <vtkCubeSource.h>
#include <vtkNamedColors.h>
#include <vtkObject.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

namespace Ithaca {

ShareCamera::ShareCamera(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ShareCameraForm)
{
    ui->setupUi(this);

    mpVtkRenderWidgetLeft  = new QVTKOpenGLNativeWidget;
    mpVtkRenderWidgetRight = new QVTKOpenGLNativeWidget;

    mpRenderWindowLeft  = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpRenderWindowRight = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
}

ShareCamera::~ShareCamera()
{
    if (mpVtkRenderWidgetLeft)
    {
        delete mpVtkRenderWidgetLeft;
        mpVtkRenderWidgetLeft = nullptr;
    }

    if (mpVtkRenderWidgetRight)
    {
        delete mpVtkRenderWidgetRight;
        mpVtkRenderWidgetRight = nullptr;
    }

    delete ui;
}

void ShareCamera::ModifiedHandler()
{
    mpVtkRenderWidgetLeft->renderWindow()->Render();
    if (mCallbackFunc)
        mCallbackFunc(true, "ShareCamera from Left");
}

void ShareCamera::Run()
{
    // Render area.
    ui->gridLayoutLeft->addWidget(mpVtkRenderWidgetLeft);
    mpVtkRenderWidgetLeft->setRenderWindow(mpRenderWindowLeft);

    ui->gridLayoutRight->addWidget(mpVtkRenderWidgetRight);
    mpVtkRenderWidgetRight->setRenderWindow(mpRenderWindowRight);

    auto pColors = vtkSmartPointer<vtkNamedColors>::New();

    // Cone
    auto pCconeSource = vtkSmartPointer<vtkConeSource>::New();
    pCconeSource->SetDirection(0.0, 1.0, 0.0);
    auto pConeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pConeMapper->SetInputConnection(pCconeSource->GetOutputPort());
    auto pConeActor = vtkSmartPointer<vtkActor>::New();
    pConeActor->SetMapper(pConeMapper);
    pConeActor->GetProperty()->SetColor(pColors->GetColor4d("Tomato").GetData());

    // Cube
    auto pCubeSource = vtkSmartPointer<vtkCubeSource>::New();
    pCubeSource->SetXLength(0.8);
    pCubeSource->SetYLength(0.8);
    pCubeSource->SetZLength(0.8);
    auto pCubeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pCubeMapper->SetInputConnection(pCubeSource->GetOutputPort());
    auto pCubeActor = vtkSmartPointer<vtkActor>::New();
    pCubeActor->SetMapper(pCubeMapper);
    pCubeActor->GetProperty()->SetColor(pColors->GetColor4d("MediumSeaGreen").GetData());

    // VTK Renderer.
    auto pLeftRenderer = vtkSmartPointer<vtkRenderer>::New();
    pLeftRenderer->AddActor(pConeActor);
    pLeftRenderer->SetBackground(pColors->GetColor3d("LightSteelBlue").GetData());

    auto pRightRenderer = vtkSmartPointer<vtkRenderer>::New();
    // Add Actor to renderer.
    pRightRenderer->AddActor(pCubeActor);
    pRightRenderer->SetBackground(pColors->GetColor3d("LightSteelBlue").GetData());

    mpVtkRenderWidgetLeft->renderWindow()->AddRenderer(pLeftRenderer);
    mpVtkRenderWidgetRight->renderWindow()->AddRenderer(pRightRenderer);

    pRightRenderer->ResetCamera();
    pLeftRenderer->ResetCamera();

    // Here we share the camera.
    pRightRenderer->SetActiveCamera(pLeftRenderer->GetActiveCamera());

    // Position the cube using the left renderer active camera.
    pLeftRenderer->GetActiveCamera()->Azimuth(60);
    pLeftRenderer->ResetCamera();

    mpVtkRenderWidgetLeft->renderWindow()->AddObserver(vtkCommand::ModifiedEvent, this, &ShareCamera::ModifiedHandler);
}

void ShareCamera::SetMessageCallback(std::function<void(bool, const std::string &)> callback)
{
    mCallbackFunc = callback;
}

} // namespace Ithaca
