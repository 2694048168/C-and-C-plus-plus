#pragma once

#include "ui_EventQtSlotConnect.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkEventQtSlotConnect.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkLookupTable.h>
#include <vtkSmartPointer.h>

#include <QWidget>
#include <functional>

/*
 * See "The Single Inheritance Approach" in this link:
 * [Using a Designer UI File in Your C++
 * Application](https://doc.qt.io/qt-5/designer-using-a-ui-file.html)
 */
QT_BEGIN_NAMESPACE

namespace Ui {
class EventQtSlotConnectForm;
}

QT_END_NAMESPACE

namespace Ithaca {

class EventQtSlotConnect : public QWidget
{
    Q_OBJECT

public:
    void Run();

    /** Get a specialised lookup table for the platonic solids.
     *
     * Since each face of a vtkPlatonicSolidSource has a different
     * cell scalar, we create a lookup table with a different colour
     * for each face.
     * The colors have been carefully chosen so that adjacent cells
     * are colored distinctly.
     *
     * @return The lookup table.
     */
    void GetPlatonicLUT();

    void SetMessageCallback(std::function<void(bool, const std::string &)> callback);

public:
    // Constructor/Destructor
    explicit EventQtSlotConnect(QWidget *parent = nullptr);
    virtual ~EventQtSlotConnect();

public slots:
    void sl_clicked(vtkObject *, unsigned long, void *, void *);

private:
    std::function<void(bool, const std::string &)> mCallbackFunc;

private:
    // Designer form
    Ui::EventQtSlotConnectForm *ui = nullptr;

    vtkSmartPointer<vtkEventQtSlotConnect> mpConnections = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow = nullptr;

    vtkSmartPointer<vtkLookupTable> mpLUT = nullptr;
};

} // namespace Ithaca