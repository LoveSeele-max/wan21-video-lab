# 来源、依赖与许可范围

本仓库记录对已有模型和推理工程的实验，不声称从零实现这些能力，也不重新授权第三方内容。

| 项目 | 使用范围 | 固定来源或记录 |
|---|---|---|
| Musubi Tuner | 生成、模型和保存路径；本仓库脚本调用其接口 | `https://github.com/kohya-ss/musubi-tuner`，commit `f8a1b03794a49239a3539015075f5123d6c07d66`；完整上游代码不随本仓库分发 |
| Wan2.1 | T2V 1.3B 及所需编码器/VAE 权重 | 下载来源与指纹见 REPRO；不分发权重，不将模型许可等同于整个工具链许可 |
| OpenAI CLIP | 采样帧与提示词的相似度 | `openai/clip-vit-base-patch32`，revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；来源和 MIT 许可依据见 `notes/week3_metrics.md`，模型卡元数据未被写成已声明 license |
| SageAttention | 独立环境安装和小张量兼容性检查 | `https://github.com/thu-ml/SageAttention`，v2.2.0 / commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`；不分发源码、二进制 wheel 或 CUDA 内核 |
| Python/Node 依赖 | 测量、视频解码、指标、表格制作 | 历史依赖快照与环境限制见 requirements.lock、REPRO 和 PUBLICATION |

第三方项目、模型、素材和依赖分别适用其权利人与许可条件。公开安装记录不代表完成未来部署、商业使用或重新分发时所需的全部许可审核。

首版没有为本仓库新增内容选定 MIT、Apache-2.0 等开放源码许可证。源代码公开可见不等于授予这些许可证；后续选择许可时应单独明确适用范围，不覆盖第三方材料。
