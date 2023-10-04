#ifndef __EDITOR_PLUGIN_INTERFACE_H__
#define __EDITOR_PLUGIN_INTERFACE_H__

#include <QObject>
#include <QString>

#include <opencv2/opencv.hpp>

class EditorPluginInterface
{
public:
    virtual ~EditorPluginInterface() {};

    // each edit operator name and implement in OpenCV is different
    // Each plugin will be a subclass of this interface, 
    // and implementation of these two pure virtual functions
    virtual QString name() = 0;
    virtual void edit(const cv::Mat& input, cv::Mat& output) = 0;
};

// unique identifier string as the ID of the interface.
#define EDIT_PLUGIN_INTERFACE_IID "weili_editorplugininterface"

// Qt¡¯s plugin system can recognize the plugins of this interface
Q_DECLARE_INTERFACE(EditorPluginInterface, EDIT_PLUGIN_INTERFACE_IID);

#endif // !__EDITOR_PLUGIN_INTERFACE_H__
