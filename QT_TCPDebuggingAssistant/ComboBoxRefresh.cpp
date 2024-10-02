#include "ComboBoxRefresh.h"

#include <QMouseEvent>

ComboBoxRefresh::ComboBoxRefresh(QWidget *parent)
    : QComboBox(parent)
{
}

void ComboBoxRefresh::mousePressEvent(QMouseEvent *event)
{
    // send signal via mouse
    if (event->button() == Qt::LeftButton)
        emit comboBox_clicked();

    return QComboBox::mousePressEvent(event);
}
