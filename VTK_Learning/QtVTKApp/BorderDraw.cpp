#include "BorderDraw.h"

#include <vtkAutoInit.h>
#include <vtkBorderRepresentation.h>
#include <vtkBorderWidget.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkLookupTable.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
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

// Constructor
BorderDraw::BorderDraw(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::BorderDrawForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    mpRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpChart        = vtkSmartPointer<vtkChartXY>::New();
    mpView         = vtkSmartPointer<vtkContextView>::New();
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
}

} // namespace Ithaca
