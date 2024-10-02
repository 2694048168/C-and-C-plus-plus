#include "SerialRefresh.h"

#include <QMouseEvent>

SerialRefresh::SerialRefresh(QWidget *parent)
    : QComboBox(parent)
{
}

void SerialRefresh::mousePressEvent(QMouseEvent *event)
{
    // send signal via mouse
    if (event->button() == Qt::LeftButton)
        emit refresh();

    return QComboBox::mousePressEvent(event);
}
