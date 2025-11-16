#pragma once

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

private:
    void Init();
    void Connects();
    void AddWidgetControl();

    void MessageTip(bool flag, const std::string &message);
    void RecordLog(bool flag, const std::string &message);

private:
    Ui::QtVTKAppClass *ui;

    QStackedWidget       *mpStackedWidget      = nullptr;
    Ithaca::DeformSphere *mpDeformSphereWidget = nullptr;
};
