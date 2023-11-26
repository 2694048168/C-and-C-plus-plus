#pragma once

#include <QtWidgets/QWidget>
#include "ui_QtJson.h"

QT_BEGIN_NAMESPACE
namespace Ui { class QtJsonClass; };
QT_END_NAMESPACE

class QtJson : public QWidget
{
    Q_OBJECT

public:
    QtJson(QWidget* parent = nullptr);
    ~QtJson();

private slots:
    void sl_ReadJson_Clicked();
    void sl_WriteJson_Clicked();

private:
    Ui::QtJsonClass* ui;
};
