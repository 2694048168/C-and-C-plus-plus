#include "EventQtSlotConnect.h"

#include <vtkAutoInit.h>
#include <vtkCommand.h>
#include <vtkNamedColors.h>
#include <vtkObject.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

namespace Ithaca {
void EventQtSlotConnect::GetPlatonicLUT()
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

void EventQtSlotConnect::SetMessageCallback(std::function<void(bool, const std::string &)> callback)
{
    mCallbackFunc = callback;
}

EventQtSlotConnect::EventQtSlotConnect(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::EventQtSlotConnectForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    mpRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpConnections  = vtkSmartPointer<vtkEventQtSlotConnect>::New();
    mpLUT          = vtkSmartPointer<vtkLookupTable>::New();
}

EventQtSlotConnect::~EventQtSlotConnect()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

void EventQtSlotConnect::Run()
{
    // Render area.
    ui->gridLayout->addWidget(mpVtkRenderWidget);
    mpVtkRenderWidget->setRenderWindow(mpRenderWindow);

    GetPlatonicLUT();

    auto pSource = vtkSmartPointer<vtkPlatonicSolidSource>::New();
    pSource->SetSolidTypeToOctahedron();

    auto pMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pMapper->SetInputConnection(pSource->GetOutputPort());
    pMapper->SetLookupTable(mpLUT);
    pMapper->SetScalarRange(0, 19);

    auto pActor = vtkSmartPointer<vtkActor>::New();
    pActor->SetMapper(pMapper);

    auto pColors = vtkSmartPointer<vtkNamedColors>::New();

    auto pRenderer = vtkSmartPointer<vtkRenderer>::New();
    pRenderer->AddActor(pActor);
    pRenderer->SetBackground(pColors->GetColor3d("SteelBlue").GetData());

    mpVtkRenderWidget->renderWindow()->AddRenderer(pRenderer);

    mpConnections->Connect(mpVtkRenderWidget->renderWindow()->GetInteractor(), vtkCommand::LeftButtonPressEvent, this,
                           SLOT(sl_clicked(vtkObject *, unsigned long, void *, void *)));
}

void EventQtSlotConnect::sl_clicked(vtkObject *, unsigned long, void *, void *)
{
    if (mCallbackFunc)
        mCallbackFunc(true, "VTK Clicked.");
}

} // namespace Ithaca
