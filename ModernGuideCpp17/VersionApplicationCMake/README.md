## 软件版本号意味着什么？

> 语义版本管理（SemVer）是一种软件版本管理方案，旨在传达版本中基本变更的含义。语义版本管理提供了一种清晰、结构化的软件版本管理方法，让开发人员更容易了解变更的影响并管理依赖关系。通过遵循 SemVer 规则，开发人员可以确保其软件以可预测的方式稳定发展。

### SemVer 使用由三部分组成的版本号： major.minor.patch.
- 主版本：当出现不兼容的 API 变动时，版本号会递增。
- 小版本：在以向后兼容的方式添加功能时递增。
- PATCH 版本：在进行向后兼容的错误修复时递增。

![version pipeline](../images/version_pipeline.gif)

#### 01 初始开发阶段
从版本 0.1.0 开始。
发布一些增量更改和错误修复：0.1.1, 0.2.0, 0.2.1, 0.3.0.
#### 02 第一个稳定版本
发布稳定版本：1.0.0.
#### 03 后续变更
补丁发布
需要对 1.0.0 进行错误修复，更新至 1.0.1。
更多错误修复：1.0.2, 1.0.3.
次要版本
1.0.3 中添加了一个向后兼容的新功能，更新至 1.1.0。
新增另一项功能：1.2.0。
新小版本中的错误修复：1.2.1, 1.2.2.
#### 重大版本
1.2.2 中引入了不向后兼容的重大变更，更新至 2.0.0。
以后的小版本和补丁更新遵循相同模式。
#### 04 特殊版本和预发布版本
**预发布版本**
用连字符和一系列以点分隔的标识符表示。
例如 alpha 版、beta 版和候选发布版：1.0.0-alpha、1.0.0-beta、1.0.0-rc.1。

**构建元数据**
用加号和一系列以点分隔的标识符表示。
示例：1.0.0+20130313144700。
