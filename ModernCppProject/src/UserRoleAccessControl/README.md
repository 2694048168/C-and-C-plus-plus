## RBAC(Role-Based Access Control),基于角色的访问控制模型，将权限管理系统划分为五个层次：
- 数据层：SQLite 数据库存储用户、角色、权限信息及关联关系
- 数据访问层：封装数据库 CRUD 操作，提供统一的数据库访问接口
- 业务逻辑层：实现权限校验、登录认证、用户管理等核心业务逻辑
- 导出接口层：通过导出宏暴露给外部调用方的 API 接口
- 调用方应用层：外部程序动态加载 DLL 并调用权限管理功能

```mermaid
flowchart TB
    subgraph 数据层
        db[("SQLite 数据库")]
    end
    
    subgraph 数据访问层
        dao[数据库访问对象]
    end
    
    subgraph 业务逻辑层
        auth[认证服务]
        perm[权限校验服务]
        mgr[用户/角色/权限管理]
    end
    
    subgraph 导出接口层
        api[导出的API接口]
    end
    
    subgraph 调用方应用层
        app[外部应用程序]
    end
    
    db --> dao
    dao --> auth
    dao --> perm
    dao --> mgr
    auth --> api
    perm --> api
    mgr --> api
    api --> app
```
