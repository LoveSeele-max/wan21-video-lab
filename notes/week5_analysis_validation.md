## Material Passport

- Origin Skill: experiment-agent / academic-research-suite
- Origin Mode: validate
- Origin Date: 2026-11-06
- Verification Status: ANALYZED
- Version Label: week5_validation_v1

## Validation Report

来源：评分者原表、两次用户说明、现有Week4性能/生成审计。总体适用性：CAUTION。
本次只复算确定性评分统计，不重新执行GPU生成，也不称完整实验已独立复现。

### 数值与来源核对

- 三包原件/补空副本哈希与上轮最终验收一致；旧Week4输入未改。
- 三份有效答案快照先完成，再打开Week5映射；本次修复了Sage来源哈希检查目标：
  包内记录对应quality_dimensions.csv，现核对该CSV并与原JSON的108项逐项比对。
  首次失败未产生结论文件，原脚本及原快照保留；修复没有更改答案或规则。
- 三个候选全部36对逐维重算；Sage仅替换8项，100项原样保留；四个对照十二维均一致。
- 原视频和106个评分代理哈希已重核；36对来源唯一、固定12单视频子集未变。
- 独立核对脚本再次由方向/身份复核FP8更好/更差类别、各维计数与分母。
- 结果没有p值、置信区间、统计显著性或等价性结论；min/max与5%是项目操作规则。
- 未定程度、无法判断、元数据缺失分开；不能以最高分填充真实未评样本，
  本次空白转换依据评分者对已完成评分的明确说明，并保留逐格来源。

### 方法检查

覆盖：11/11，以下为适用性检查，不是声称所有偏差都不存在。

| 类型 | 级别 | 核对结果与边界 |
|---|---|---|
| Simpson's paradox | CAUTION | 三个维度和两套环境分别报告，不合成总体胜率；未验证提示词子组推广。 |
| Ecological fallacy | NOTE | 结果只描述固定视频对，不外推所有人物、提示词或模型。 |
| Berkson's paradox | CAUTION | 固定12提示词是窄任务集，非总体随机抽样；不声称无选择偏差。 |
| Collider bias | NOTE | 未根据质量或自动指标筛除样本，未拟合协变量模型。 |
| Base rate neglect | NOTE | 无诊断敏感度/阳性预测值推断；每维保留预定36分母。 |
| Regression to the mean | CAUTION | Sage补评针对旧缺口，已看过视频；不把补全记为质量提升。 |
| Survivorship bias | NOTE | 全36样本保留，生成失败/无法判断单列；72个FP8视频均首次成功。 |
| Look-elsewhere effect | NOTE | 三配置三维全报告，Top5仅作异常对照；无显著性检验。 |
| Garden of forking paths | CAUTION | 保留原预登记、补充、原答卷和解盲前快照；程度未定不补档。 |
| Correlation versus causation | CAUTION | 速度只比较组内；绝对人物缺陷不自动归因于优化新增。 |
| Reverse causality | NOTE | 无横断面回归或方向性因果模型；不作反向因果解释。 |

### 复现范围

验证状态ANALYZED；评分计数已本机重复核算，源视频仅核对哈希，未由助手重新观看代评。
GPU性能实验未重跑，干净环境独立复现和VAE峰值内部归属仍未验证。
完整审计入口为`notes/blind_private/week5-analysis-20261106/analysis_audit.json`和
`report_verification.json`；私人映射、原评分及细节不自动公开。
