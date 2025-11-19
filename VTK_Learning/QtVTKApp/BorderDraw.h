#pragma once

#include "ui_BorderDraw.h"

#include <QVTKOpenGLNativeWidget.h>
#include <vtkBorderWidget.h>
#include <vtkChartXY.h>
#include <vtkContextView.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
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
class BorderDrawForm;
}

QT_END_NAMESPACE

namespace Ithaca {

namespace {
class vtkCustomBorderWidget : public vtkBorderWidget
{
public:
    static vtkCustomBorderWidget *New();

    vtkTypeMacro(vtkCustomBorderWidget, vtkBorderWidget);

    static void EndSelectAction(vtkAbstractWidget *w);

    vtkCustomBorderWidget();

    void SetMessageCallback(std::function<void(bool, const std::string &)> &callback);

public:
    std::function<void(bool, const std::string &)> mCallbackFunc;
};

vtkStandardNewMacro(vtkCustomBorderWidget);
} // namespace

class BorderDraw : public QWidget
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
    explicit BorderDraw(QWidget *parent = nullptr);
    virtual ~BorderDraw();

private:
    // Designer form
    Ui::BorderDrawForm *ui = nullptr;

    QVTKOpenGLNativeWidget *mpVtkRenderWidget = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> mpRenderWindow = nullptr;

    vtkSmartPointer<vtkCustomBorderWidget> mpBorderWidget = nullptr;

    vtkSmartPointer<vtkLookupTable> mpLUT = nullptr;
};

} // namespace Ithaca