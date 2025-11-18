#include "BarChart.h"

#include <vtkAutoInit.h>
#include <vtkAxis.h>
#include <vtkIntArray.h>
#include <vtkNamedColors.h>
#include <vtkPlot.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkTextProperty.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2);
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

#include <array>
#include <string>

namespace Ithaca {

// Monthly circulation data.
int data_2008[] = {10822, 10941, 9979, 10370, 9460, 11228, 15093, 12231, 10160, 9816, 9384, 7892};
int data_2009[] = {9058, 9474, 9979, 9408, 8900, 11569, 14688, 12231, 10294, 9585, 8957, 8590};
int data_2010[] = {9058, 10941, 9979, 10270, 8900, 11228, 14688, 12231, 10160, 9585, 9384, 8590};

// Constructor
BarChart::BarChart(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::BarChartForm)
{
    ui->setupUi(this);

    mpVtkRenderWidget = new QVTKOpenGLNativeWidget;

    mpRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    mpChart        = vtkSmartPointer<vtkChartXY>::New();
    mpView         = vtkSmartPointer<vtkContextView>::New();
}

BarChart::~BarChart()
{
    if (mpVtkRenderWidget)
    {
        delete mpVtkRenderWidget;
        mpVtkRenderWidget = nullptr;
    }

    delete ui;
}

void BarChart::Run()
{
    // Render area.
    ui->gridLayout->addWidget(mpVtkRenderWidget);
    mpVtkRenderWidget->setRenderWindow(mpRenderWindow);

    auto       pColors         = vtkSmartPointer<vtkNamedColors>::New();
    vtkColor3d backgroundColor = pColors->GetColor3d("Seashell");
    vtkColor3d axisColor       = pColors->GetColor3d("Black");
    vtkColor3d titleColor      = pColors->GetColor3d("MidnightBlue");

    // Set various properties.
    vtkAxis *xAxis = mpChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->SetTitle("Monthly");
    xAxis->GetTitleProperties()->SetColor(axisColor.GetData());
    xAxis->GetTitleProperties()->SetFontSize(16);
    xAxis->GetTitleProperties()->ItalicOn();
    xAxis->GetLabelProperties()->SetColor(axisColor.GetData());
    xAxis->SetGridVisible(true);
    xAxis->GetGridPen()->SetColor(pColors->GetColor4ub("Black"));

    vtkAxis *yAxis = mpChart->GetAxis(vtkAxis::LEFT);
    yAxis->SetTitle("Circulation");
    yAxis->GetTitleProperties()->SetColor(axisColor.GetData());
    yAxis->GetTitleProperties()->SetFontSize(16);
    yAxis->GetTitleProperties()->ItalicOn();
    yAxis->GetLabelProperties()->SetColor(axisColor.GetData());
    yAxis->SetGridVisible(true);
    yAxis->GetGridPen()->SetColor(pColors->GetColor4ub("Black"));

    mpChart->SetTitle("Circulation 2008, 2009, 2010");
    mpChart->GetTitleProperties()->SetFontSize(24);
    mpChart->GetTitleProperties()->SetColor(titleColor.GetData());
    mpChart->GetTitleProperties()->BoldOn();

    // Set up a 2D scene, add an XY chart to it
    mpView->SetRenderWindow(mpRenderWindow);
    mpView->GetRenderer()->SetBackground(backgroundColor.GetData());
    //mpView->GetRenderWindow()->SetSize(640, 480);
    mpView->GetScene()->AddItem(mpChart);

    // Create a table with some points in it...
    auto pTable = vtkSmartPointer<vtkTable>::New();

    auto pArrMonth = vtkSmartPointer<vtkIntArray>::New();
    pArrMonth->SetName("Month");
    pTable->AddColumn(pArrMonth);

    auto pArr2008 = vtkSmartPointer<vtkIntArray>::New();
    pArr2008->SetName("2008");
    pTable->AddColumn(pArr2008);

    auto pArr2009 = vtkSmartPointer<vtkIntArray>::New();
    pArr2009->SetName("2009");
    pTable->AddColumn(pArr2009);

    auto pArr2010 = vtkSmartPointer<vtkIntArray>::New();
    pArr2010->SetName("2010");
    pTable->AddColumn(pArr2010);

    pTable->SetNumberOfRows(12);
    for (int i = 0; i < 12; i++)
    {
        pTable->SetValue(i, 0, i + 1);
        pTable->SetValue(i, 1, data_2008[i]);
        pTable->SetValue(i, 2, data_2009[i]);
        pTable->SetValue(i, 3, data_2010[i]);
    }

    // Add multiple line plots, setting the colors etc.
    vtkPlot *pLine = nullptr;

    pLine = mpChart->AddPlot(vtkChart::BAR);
    pLine->SetInputData(pTable, 0, 1);
    auto rgba = pColors->GetColor4ub("YellowGreen");
    pLine->SetColor(rgba[0], rgba[1], rgba[2], rgba[3]);

    pLine = mpChart->AddPlot(vtkChart::BAR);
    pLine->SetInputData(pTable, 0, 2);
    rgba = pColors->GetColor4ub("Salmon");
    pLine->SetColor(rgba[0], rgba[1], rgba[2], rgba[3]);

    pLine = mpChart->AddPlot(vtkChart::BAR);
    pLine->SetInputData(pTable, 0, 3);
    rgba = pColors->GetColor4ub("CornflowerBlue");
    pLine->SetColor(rgba[0], rgba[1], rgba[2], rgba[3]);

    mpView->GetRenderWindow()->SetMultiSamples(0);
    mpVtkRenderWidget->setRenderWindow(mpView->GetRenderWindow());
}

} // namespace Ithaca
