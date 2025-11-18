#pragma once

#include "BarChart.h"
#include "BorderDraw.h"
#include "DeformSphere.h"
#include "ui_QtVTKApp.h"

#include <QStackedWidget>
#include <QtWidgets/QMainWindow>
#include <string>

QT_BEGIN_NAMESPACE

namespace Ui {
class QtVTKAppClass;
};

QT_END_NAMESPACE

class QtVTKApp : public QMainWindow
{
    Q_OBJECT

public:
    QtVTKApp(QWidget *parent = nullptr);
    ~QtVTKApp();

public slots:
    void sl_DeformSphere();
    void sl_BarChart();
    void sl_BorderDraw();

private:
    void Init();
    void Connects();
    void AddWidgetControl();

    void MessageTip(bool flag, const std::string &message);
    void RecordLog(bool flag, const std::string &message);

    std::string GetVersion();

private:
    Ui::QtVTKAppClass *ui;

    QStackedWidget       *mpStackedWidget      = nullptr;
    Ithaca::DeformSphere *mpDeformSphereWidget = nullptr;
    Ithaca::BarChart     *mpBarChartWidget     = nullptr;
    Ithaca::BorderDraw   *mpBorderDrawWidget   = nullptr;

    int mMajor = 0;
    int mMinor = 1;
    int mBuild = 1;
};
