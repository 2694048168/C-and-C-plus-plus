#include "BorderDraw.h"

#include <vtkAutoInit.h>
#include <vtkBorderRepresentation.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkNamedColors.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkVersion.h>
#include <vtkWidgetCallbackMapper.h>
#include <vtkWidgetEvent.h>

#include <iostream>
#include <string>

VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

namespace Ithaca {

void BorderDraw::GetPlatonicLUT()
{
    mpLUT->SetNumberOfTableValues(20);
    mpLUT->SetTableRange(0.0, 19.0);
    mpLUT->Build();
    mpLUT->SetTableValue(0, 0.1, 0.1, 0.1);
    mpLUT->SetTableValue(1, 0, 0, 1);
    mpLUT->SetTableValue(2, 0, 1, 0);
    mpLUT->SetTableValue(3, 0, 1, 1);
    mpLUT->SetTableValue(4, 1, 0, 0);
    mpLUT->SetTableValue(5, 1, 0, 1);
    mpLUT->SetTableValue(6, 1, 1, 0);
    mpLUT->SetTableValue(7, 0.9, 0.7, 0.9);
    mpLUT->SetTableValue(8, 0.5, 0.5, 0.5);
    mpLUT->SetTableValue(9, 0.0, 0.0, 0.7);
    mpLUT->SetTableValue(10, 0.5, 0.7, 0.5);
    mpLUT->SetTableValue(11, 0, 0.7, 0.7);
    mpLUT->SetTableValue(12, 0.7, 0, 0);
    mpLUT->SetTableValue(13, 0.7, 0, 0.7);
    mpLUT->SetTableValue(14, 0.7, 0.7, 0);
    mpLUT->SetTableValue(15, 0, 0, 0.4);
    mpLUT->SetTableValue(16, 0, 0.4, 0);
    mpLUT->SetTableValue(17, 0, 0.4, 0.4);
    mpLUT->SetTableValue(18, 0.4, 0, 0);
    mpLUT->SetTableValue(19, 0.4, 0, 0.4);
}

// Constructor
BorderDraw::BorderDraw(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::BorderDrawForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    mpRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpBorderWidget = vtkSmartPointer<vtkCustomBorderWidget>::New();
    mpLUT          = vtkSmartPointer<vtkLookupTable>::New();
}

BorderDraw::~BorderDraw()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

void BorderDraw::Run()
{
    // Render area.
    ui->gridLayout->addWidget(mpVtkRenderWidget);
    mpVtkRenderWidget->setRenderWindow(mpRenderWindow);

    vtkNew<vtkNamedColors> colors;

    GetPlatonicLUT();

    vtkNew<vtkPlatonicSolidSource> source;
    source->SetSolidTypeToDodecahedron();

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(source->GetOutputPort());
    mapper->SetLookupTable(mpLUT);
    mapper->SetScalarRange(0, 19);

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);

    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->GetActiveCamera()->Elevation(30.0);
    renderer->GetActiveCamera()->Azimuth(180.0);
    renderer->ResetCamera();
    renderer->SetBackground(colors->GetColor3d("SteelBlue").GetData());

    // Connect VTK with Qt.
    mpVtkRenderWidget->renderWindow()->AddRenderer(renderer);
    // Add a border widget to the renderer.
    mpBorderWidget->SetInteractor(mpVtkRenderWidget->interactor());
    mpBorderWidget->CreateDefaultRepresentation();
    mpBorderWidget->SelectableOff();
    mpBorderWidget->On();
}

void BorderDraw::SetMessageCallback(std::function<void(bool, const std::string &)> callback)
{
    mpBorderWidget->SetMessageCallback(callback);
}

namespace {

vtkCustomBorderWidget::vtkCustomBorderWidget()
{
    this->CallbackMapper->SetCallbackMethod(vtkCommand::MiddleButtonReleaseEvent, vtkWidgetEvent::EndSelect, this,
                                            vtkCustomBorderWidget::EndSelectAction);
}

void vtkCustomBorderWidget::SetMessageCallback(std::function<void(bool, const std::string &)> &callback)
{
    mCallbackFunc = callback;
}

void vtkCustomBorderWidget::EndSelectAction(vtkAbstractWidget *w)
{
    vtkBorderWidget *borderWidget = dynamic_cast<vtkBorderWidget *>(w);

    // Get the actual box coordinates/planes.
    // vtkNew<vtkPolyData> polydata;

    // Get the bottom left corner.
    auto lowerLeft  = static_cast<vtkBorderRepresentation *>(borderWidget->GetRepresentation())->GetPosition();
    auto messageStr = "Lower left: " + std::to_string(lowerLeft[0]) + " " + std::to_string(lowerLeft[1]);

    auto upperRight  = static_cast<vtkBorderRepresentation *>(borderWidget->GetRepresentation())->GetPosition2();
    auto messageStr2 = "Upper right: " + std::to_string(lowerLeft[0] + upperRight[0]) + " "
                     + std::to_string(lowerLeft[1] + upperRight[1]);

    if (((vtkCustomBorderWidget *)borderWidget)->mCallbackFunc)
        ((vtkCustomBorderWidget *)borderWidget)->mCallbackFunc(true, messageStr + " " + messageStr2);

    vtkBorderWidget::EndSelectAction(w);
}

} // namespace

} // namespace Ithaca
