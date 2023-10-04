#ifndef __ROTATE_PLUGIN_H__
#define __ROTATE_PLUGIN_H__

#include <QObject>
#include <QtPlugin>

#include "rotateplugin_global.h"

class RotatePlugin :public QObject, public EditorPluginInterface
{
private:
	Q_OBJECT
		Q_PLUGIN_METADATA(IID EDIT_PLUGIN_INTERFACE_IID);
	Q_INTERFACES(EditorPluginInterface);

public:
	QString name() override;
	void edit(const cv::Mat& input, cv::Mat& output) override;
};

#endif // !__ROTATE_PLUGIN_H__
