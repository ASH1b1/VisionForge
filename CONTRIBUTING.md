# Contributing to VisionForge

[简体中文](#中文) | [English](#english)

## 中文

欢迎提交 Issue 与 Pull Request。

### 开发约定

- 新功能写进对应的 Controller / Model / Exporter / Importer，不要把逻辑堆进 `MainWindow`。
- 工程数据写入走 `ProjectDocument`。检测后处理走 `src/utils/annotation_processor.py`。
- 提交前请运行：

```powershell
conda run -n visionforge python -m pytest src/tests/ -v --tb=short
```

### Developer Certificate of Origin (DCO)

向本仓库提交代码，即表示你同意按 Apache License 2.0 授权你的贡献（`sam3_src/` 等第三方目录除外，那些文件保持原许可）。

每个 commit 必须包含：

```
Signed-off-by: Your Name <your.email@example.com>
```

使用 `git commit -s` 可自动添加。`Signed-off-by` 表示你核证了 [DCO 1.1](https://developercertificate.org/)：

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## English

Issues and pull requests are welcome.

### Development rules

- Put new behavior in the owning Controller / Model / Exporter / Importer, not in `MainWindow`.
- Persist project data through `ProjectDocument`. Run detection post-processing through `src/utils/annotation_processor.py`.
- Before you submit:

```powershell
conda run -n visionforge python -m pytest src/tests/ -v --tb=short
```

### Developer Certificate of Origin (DCO)

By contributing, you license your work under Apache License 2.0 (except third-party trees such as `sam3_src/`, which keep their original licenses).

Every commit must include:

```
Signed-off-by: Your Name <your.email@example.com>
```

`git commit -s` adds this line. It certifies the [DCO 1.1](https://developercertificate.org/) text quoted above.
