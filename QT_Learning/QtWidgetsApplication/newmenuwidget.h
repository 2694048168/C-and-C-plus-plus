#pragma once

#include <QWidget>
#include "ui_newmenuwidget.h"

class newMenuWidget : public QWidget
{
	Q_OBJECT

public:
	newMenuWidget(QWidget *parent = nullptr);
	~newMenuWidget();

private:
	Ui::newMenuWidgetClass ui;
};
