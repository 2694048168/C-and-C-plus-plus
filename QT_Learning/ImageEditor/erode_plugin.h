#include <QObject>
#include <QtPlugin>

#include "editor_plugin_interface.h"

class ErodePlugin : public QObject, public EditorPluginInterface
{
private:
	//Q_OBJECT
	Q_PLUGIN_METADATA(IID EDIT_PLUGIN_INTERFACE_IID);
	Q_INTERFACES(EditorPluginInterface);

public:
	QString name() override;
	void edit(const cv::Mat& input, cv::Mat& output) override;
};
