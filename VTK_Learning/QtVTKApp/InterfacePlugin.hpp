#pragma once

#include <QMap>
#include <QVariantMap>

namespace Ithaca {

template<typename Interface>
QMap<Interface *, QVariantMap> LoadPlugins(const QString &strPluginDir)
{
    QMap<Interface *, QVariantMap> loadedPlugins;

    QDir pluginDir(strPluginDir);
    for (const QString &fileName : pluginDir.entryList(QDir::Files))
    {
        if (!QLibrary::isLibrary(fileName))
            continue;

        QString       file = pluginDir.absoluteFilePath(fileName);
        QPluginLoader loader(file);
        Interface    *interface = qobject_cast<Interface *>(loader.instance());
        if (interface)
        {
            QVariantMap temp                      = loader.metaData().toVariantMap();
            temp[QString::fromLatin1("fileName")] = QFileInfo(fileName).baseName();
            loadedPlugins[interface]              = temp;
        }
        else
        {
            qDebug() << loader.errorString();
            loader.unload();
        }
    }
    return loadedPlugins;
}

template<typename Interface>
QPair<Interface *, QVariantMap> loadPlugin(const QString &pluginFile, QString *error = nullptr)
{
    if (!QFile::exists(pluginFile))
    {
        Q_ASSERT(0);
        return qMakePair(nullptr, QVariantMap());
    }
    QPluginLoader loader(pluginFile);
    Interface    *interface = qobject_cast<Interface *>(loader.instance());
    if (error)
        *error = loader.errorString();
    if (interface)
    {
        return qMakePair(interface, loader.metaData().toVariantMap());
    }
    else
    {
        loader.unload();
        return qMakePair(nullptr, QVariantMap());
    }
}

} // namespace Ithaca
