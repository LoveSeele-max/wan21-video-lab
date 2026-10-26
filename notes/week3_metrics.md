# Week 3 自动指标协议 v1

2026-10-19，计算结果前登记。自动指标用于反例测试，不代替人工评分。

## 权重与许可（下载前记录）

- 模型：`openai/clip-vit-base-patch32`。
- 固定 revision：`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，由官方 Hugging Face 模型 API 查询。
- 来源：https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
- 许可依据：OpenAI CLIP 官方仓库 MIT LICENSE：https://github.com/openai/CLIP/blob/main/LICENSE 。保留版权和许可文本。HF 模型卡本身没有单独的 license 元数据，不能将 MIT 标签误写为已在该模型卡声明。
- 模型卡：https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/README.md 。模型定位为研究，模型卡列有部署、英语以外文本和细粒度识别的局限。本次仅离线英语提示词实验。
- 下载脚本将模型卡、MIT 许可原文、权重 SHA256 与本记录的下载前哈希一起保留。
- 缓存：`local/models/clip-vit-base-patch32`，不使用 C 盘默认 HF 缓存；只下载 PyTorch 权重和必要配置，不下载 TF/Flax 副本。

## 固定计算口径

- PyAV 以显示顺序解码为 RGB uint8；严格检查 81 帧、832×480、16 fps。原视频不重编码、不修改。
- SSIM：OpenCV RGB2GRAY，原分辨率，相邻 80 对；scikit-image 0.25.2 的 structural_similarity，data_range=255，win_size=11，gaussian_weights=True，sigma=1.5，use_sample_covariance=False，K1=0.01，K2=0.03；报告均值和最小值，并保存全部逐帧对值。
- CLIP：固定模型原始 processor 的缩放、中心裁剪、归一化；第 0/4/.../80 帧，共 21 帧。FP32，eval、inference_mode，每批最多 7 帧。归一化后的 image_embeds 与 text_embeds 点积为余弦相似度，不用 softmax、不乘 logit_scale；报告均值和最低值，保存逐帧值及实际索引。
- 默认 CPU 计算 CLIP，避免占用本机推理显存；允许显式指定 cuda，但须另输出目录并记录设备，不能混写。
- 每条记录包含输入视频 SHA256；清单对应哈希不符时记录 error，不静默跳过。输出 CSV、逐视频 JSONL、运行元数据与依赖版本。失败行保留并非零退出。

## 探针构造（结果前冻结）

- 来源：P01_s42、P04_s42、P07_s42。它们是原始基线，并非已经人工确认的“好视频”。
- 每个来源保留原始视频作为正常对照；另构造四段，共 12 段：
  - static：首帧重复 81 次。
  - flicker：**0-based 第 30 到 40 帧，含端点，共 11 帧**，依次乘 1.15/0.85，截断到 0–255、四舍五入。
  - blur：同一帧区间，OpenCV GaussianBlur，核 19×19，sigmaX=sigmaY=3，BORDER_REFLECT_101。
  - local_warp：同一区间，在预先查看第 35 帧后固定的主体区域做局部水平正弦位移；位置、幅值和频率写入 probe manifest。海浪使用中心岩石/海浪区域，不声称是人物主体。区域外像素不变。
- 探针用 FFV1、bgr0、Matroska 无损保存；编码后重新解码逐帧核对像素，排除有损二次压缩造成的混淆。原 MP4 和探针容器不同，但输入到指标的 RGB 像素明确可校验。
- 动作不符：P02_s42 原视频配 P01 文本；另外保留 P02 原文本分数作为配对参照。若 P02 本身未生成人物点头，不能称为已验证的“动作不符”反例，标为文本替换候选，待 Day5 人工检查。

## 判定规则（先于结果）

对四个标量分别计算 36 条基线的最小值 lo、最大值 hi，边缘宽度 `0.05*(hi-lo)`：

1. 探针值严格落在 `[lo,hi]` 之外：按计划记“能区分”（仅指范围外异常）。
2. 落在闭区间内且到任一端点距离不超过边缘宽度：记“部分区分”。
3. 其余：记“不能区分”。若基线范围为零，与基线相等记“不能区分”。

这不是统计显著性检验，也不是通用好坏阈值。另报相对来源的差值、范围外的方向。静止 SSIM=1 即使被标为范围外“能区分”，也不能解读成质量退化被正确识别。三个来源分别列结果，不用多数票合成指标能力。

Day4 仅计算并解释指标；探针混入人工盲评在 Day5，完成前不勾选。
