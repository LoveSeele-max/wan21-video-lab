# Week 5 执行日志

日期：2026-11-01，Asia/Shanghai。

## Day 1：预登记补充与 FP8 自动指标

- 用户确认 D4 选项 A：阶段峰值下降可记为有条件资源收益，但整次峰值未下降；只有其他阶段峰值也下降后才可能转为整次显存收益。未验证组合效果。
- Sage 补评规模确定为 10 个唯一视频对：T001、T002、T003、T010、T032、T038 六个争议对（覆盖八项未解决维度答案），另加四个从明确回答中随机抽取的对照对。每个视频对只出现一次。
- 在制作评分包、查看任何 Week5 评分答案前写定 `notes/week5_prereg_addendum.md`。Week4 原预登记、原答卷和原结论未修改。
- 两个输出目录在运行前均不存在。FP8 dry-run 均通过，验证候选/基线视频哈希、冻结协议、指标实现和依赖版本。
- 普通 FP8 使用本机 `local/env/python.exe`、CPU / FP32 执行；36/36 成功，0 错误，2880 个 SSIM 配对值、756 个 CLIP 帧值，脚本状态 `ok`，运行约 163.57 秒。
- scaled FP8 同环境执行；36/36 成功，0 错误，2880 个 SSIM 配对值、756 个 CLIP 帧值，脚本状态 `ok`，运行约 149.36 秒。
- 汇总及各指标差值绝对值最大的五个样本见 `notes/week5_metrics_fp8.md`；完整逐视频数据、配对表、排名和 run 元数据保存在两个 `notes/week5_metrics_*` 目录中。
- 自动指标只作异常线索，不替代人工评分，不据此选择、排除或替换样本；人工评分结束后再对照。

Day 1 出口完成，后续转入 Day 2 匿名评分包制作与验收。

## Day 2：匿名评分包

- Sage 补评包：10 个唯一来源对，6 个争议对覆盖 8 项未解决维度答案，4 个随机对照对；随机种子及来源映射仅存私人目录。评分者侧编号为 T101–T110。每段 81 帧，16 fps，H.264/yuv420p；源视频 SHA256、左右方向和代理帧抽查通过。
- 普通 FP8 与 scaled FP8 各制作 36 对 + 12 段单视频。三者均有空白评分工作簿、CSV 备用表、冻结评分标准和说明文件。
- 三份工作簿用 Artifact Tool 创建；验证观看状态计数公式从 0→1→0、评分输入格为空、下拉验证存在、公式错误扫描为 0，并检查渲染。
- 普通 FP8 和 scaled FP8 的 48 个视频均通过来源覆盖、配对方向、提示词、解码及代理误差核验。
- 三份 ZIP 均只含评分者所需文件，无私人映射；ZIP CRC、文件清单、工作簿空白评分格及隐私扫描通过。补评包另通过随机种子重放和 6+4 配对数量核对。
- 补评 ZIP SHA256：`c39c8a96fd83cc5e56565f0a4bd08a95a544695fdbff3a2991269b139817263d`。
- 普通 FP8 ZIP SHA256：`4a58f6cf7c361d5c34fd9d9a1ece411f19d626bf07bafb90191753fa1a08f3c1`。
- scaled FP8 ZIP SHA256：`f695016023f8fb7fb931a00d24a50b5bbe2ec151532819289c7e730922aeb5da`。
- 评分顺序：先发 Sage 补评与普通 FP8；scaled FP8 包已制作但不得早于 **2026-11-02** 开始评分，以满足至少一天间隔。

Day 2 打包和验收完成。后续等待 R01 返回评分表；收到后先复制归档并核验原件 SHA256，再检查冲突，之后才解盲。

## Day 3–4：评分回收与补空（2026-11-05）

- 用户确认三个包均已评分完成，并明确配对空白评分选项代表无差异、单视频空白分数代表 2 分。
- 三份原件先按 SHA256 归档；原路径和 `original.xlsx` 均未重新保存。按授权生成独立补填副本，Sage / 普通 FP8 / scaled FP8 分别补 60 / 46 / 213 格，共 319 格。
- 输入身份、已有答案和备注逐格保留；会话、观看状态、查看方式及问题时间没有代填。单视频不使用的程度格仍留空。
- 三份最终副本已通过评分值、原生格式/公式/下拉/窗格/表格/行列尺寸、ZIP CRC 核对，并重新渲染检查变化区域。工具导出时的保留差异已从原件恢复，最终报告为 `completed_workbook_verification_final.json`。
- 已固定最终解盲前答案快照；未读取三个 Week5 映射。Sage 无方向/程度冲突；普通 FP8 有 5 项、scaled FP8 有 2 项原填冲突，等待一次澄清，不能自动改判。
- 实际评分日期缺失，两个 FP8 包至少一天间隔尚未核实；不据文件修改时间猜测。
- 详细回收清单、证据和后续步骤见 `notes/week5_rating_review.md`。Day 3–4 仅完成回收、补空和解盲前审查，尚未完成澄清与解盲；Day 5 结论卡未生成。

## Day 3–6：一次澄清、解盲及报告（2026-11-06）

- 用户说明“按照已填方向为准，评分相隔两天”。七项保留原方向：T022 无法判断，另六项方向确定而程度未定。两包间隔满足至少一天；来源为评分者陈述，具体日期未知。缺失的观看/会话字段未补造。
- `scripts/analyze_week5_ratings.py prepare` 先固定三份有效答案及输入哈希，随后 `analyze` 才读取映射。首次分析因 Sage 来源哈希校验对象错误而停止，尚未输出结论；包内哈希实际对应旧 `quality_dimensions.csv`。保留原脚本，改为核对该 CSV 并逐项核对旧 JSON 的 108 个答案；答案快照、评分和原规则没有改变。审计记录原/新脚本哈希及修复原因。
- Sage 只替换八项原未解决维度，其他一百项原样保留；四个对照共十二项类别一致，无方向相反或程度差至少两档。三维完整相对答案均 36/36、明显变差均 0、轻微变差 0/1/0；质量数值门槛达到。关键记录仍不完整，综合结论不足。
- 普通 FP8 结构/时序/符合度完整相对答案 36/31/36；三维各有一项原表明确填写的明显变差，来自同一个唯一视频对。保留时序四项程度未定、一项无法判断。当前任务不推荐。
- scaled FP8 完整相对答案 35/35/36；时序两项、符合度一项原表明确填写的明显变差，共两个唯一视频对；结构/时序各一项确定更差但程度未定。当前任务不推荐。
- 原文字未分别解释所有退化维度，报告保留该局限。助手未复看代评，不将配对判断说成已独立确认的所有维度退化；十二段单视频最高分不抵消其他来源配对的退化。
- 源视频 118 个去重路径及 106 个评分代理哈希已核对。`report_week5_results.py` 再核对输入哈希、FP8 方向/身份与九组维度计数；生成三张结果卡、质量汇总、验证报告、权衡表与阶段显存图。图已目视检查，原始数值另存 CSV。
- 两种 FP8 各四指标 Top5 与全部人工答案对照，共 40 行：普通 FP8 覆盖 12 个唯一来源，与明显退化来源交集 1 个；scaled 覆盖 16 个，交集 0。只是异常线索，不筛样本或另设事后阈值。
- 报告绘图库在现有两个 Python 中均缺失；在独立 `<USER_HOME>/.cache/week5-report-plotting/` 安装绘图依赖后生成图，未改项目推理环境。最初导入失败未写任何报告文件。
- 总览、执行清单、Week5 计划及 REPRO 已同步；未来 `performance_rules_v2.md` 使用固定六个顺序平衡的相邻配对区组，不以合并 min/max 消除已有交集。仅供新实验事前引用，不重判本轮。
- 最终本机检查：七份报告产物哈希一致，十二份当前文档共 64 个本地链接有效，两份分析/报告脚本语法编译通过；九组维度计数核对与 40 行指标对照完成。图已检查无裁切或标签重叠；没有据此宣称 GPU 性能独立复现。

Day1–6 已完成。Day7 尚未开始服务器测量；基线 manifest 已提供 P01/s42 latent 路径和历史哈希，当前文件存在性、哈希、dtype/shape 及 GPU0 占用须先现场核实。下一步见 `notes/week5_vae_preflight.md`。没有修改 VAE 算法，没有重跑既有 total/stages 或生成，Week5 尚未收尾。

## Day7：服务器输入预检回收（2026-11-06）

- 用户回传版本/文件/GPU 检查输出：固定上游 commit 匹配，P01/s42 latent 存在、约 8.0M，SHA256 `45b483abe74e4cc29943b15c40931aa11ff265d652abf964132636303f0def16` 与历史 manifest 一致。
- GPU0 当时 600 MiB、利用率 0%，可见进程均为桌面相关，未见推理计算任务；GPU1 为 20 MiB、0%。占用只代表该快照，不替代正式测量前检查。服务器显示 11:27:18，其时区未单独核实。
- 驱动显示 580.126.20；nvidia-smi 的 CUDA 13.0 不作为 Python torch/CUDA build 的证据。来源为用户服务器输出，助手未远程执行，也未在本机独立读取 latent。
- 下一步使用主环境将 latent 只读加载至 CPU，检查 dtype/shape/finite 和 metadata。尚未运行 VAE 解码，固定输入登记及 Day7 定位未完成。

### latent 内容回收与仅解码脚本准备

- 用户已回传 latent 为 FP32，shape `[16,21,60,104]`，finite=true；min/max=-2.86393404006958/3.138010263442993。seed、prompt、guidance、尺寸、帧数和步数均与历史 manifest 匹配；已另存 `notes/week5_vae_fixed_input.json` 并勾选输入登记，来源仍为服务器用户回传。
- 新建 `scripts/vae_decode_measure.py`，调用固定上游 latent-only 分支，解码前后同步并在区间开始重置峰值。耗时包括 VAE 加载、传输、解码及 FP32/CPU 转换，不含 MP4 编码；当前不打开 memory history。只包装，不修改 VAE 或原 bench.py。
- 本机项目 Python dry-run 通过；模拟检查同步/重置顺序、单次 decode 限制、峰值单位、cache 参数仅增加一个开关及语法均通过。这些不是服务器 GPU 验证。
- 脚本校验固定上游/输入/环境，使用新目录并保留异常结果。已准备 scp、tmux 和单次 baseline 测量命令；尚未上传执行、没有峰值结果，暂不勾选仅解码完成或内部分配定位。

### 仅解码服务器摘要回收

- 用户回传 `outputs/week5-vae/decode-baseline-01/result.json` 的终端摘要，status=ok；耗时 10.99382509899442 秒，allocated 峰值 15412987904 字节（14698.970703125 MiB），reserved 19581108224 字节（18674 MiB）。
- 输出张量为 CPU FP32 `[3,81,480,832]`，视频路径已回传；after allocated=8519680 字节，reserved 仍高，不据单次读数断言泄漏或 feat_cache 占用。
- 本机对照原 `fp8_stages_audit.json` 基线 VAE 峰值 14698.353515625 MiB，差约 0.6171875 MiB / 0.004199%；没有出现明显峰值不匹配。两次时间的边界/上下文不同，不比较为优化收益。
- 用户摘要另存 `notes/week5/vae-baseline-01-user-return.json`，明确不是原服务器 result.json。助手尝试 BatchMode scp 只读取回，但 SSH 无交互认证失败，未取得原 JSON 或视频；未远程执行计算。
- 下一步用户在本机 scp 取回整个运行目录后，核验原始身份/哈希和视频完整解码。memory-history 探针与内部分配归属尚未执行，Week5 未完成，不重跑已成功轮次。

### 仅解码原件验收与快照探针准备

- 用户已传回 input.json/result.json/MP4。新增 `audit_vae_decode.py`，核验原始字段、固定环境/输入和测量脚本身份；视频哈希匹配服务器记录及原 P01/s42 基线，且本机完整解码 81 帧、832×480、16 fps，无全黑帧。
- 首次源码原始字节哈希检查中止：本机 Windows 工作树 CRLF，服务器为 LF。核对服务器哈希与固定 commit Git blob 完全相同，本机仅换行差异；校正核验对象后审计通过。未改任何输入或上游源码，早期未产生成功审计。机器结果 `notes/week5/vae-baseline-01-local-audit.json` 保留双方哈希与限制。
- VAE/latent 本体仍未传回，其内容来源边界不变；两个 JSON 未单独提供服务器哈希，不称已经独立核对跨机 JSON 哈希。输出字节相同不等于全部样本质量通过或已定位内部来源。
- 新增 `vae_memory_probe.py`，复用原基本测量入口且固定其哈希，运行时与原基线核对输入/环境/VAE 身份。allocator history 上限 200000 条，conv2/decoder 边界保存形状和去重 storage 缓存字节；不改数值运算，不在 hook 内额外同步，不持有额外 GPU 张量引用。
- 本机同版 torch 2.9.1+cu128 的实际源码/签名核对完成；dry-run 和模拟测试覆盖缓存 storage 去重、快照保存/关闭、异常原样保留、21 次 decoder/44 个边界及成功/异常 hook 清理。基础测量脚本未改，真实 GPU 探针未执行。
- 已建立 `notes/week5_vae_peak.md` 阶段记录，关键三问仍未回答；计划仅勾选输入登记、基本测量验收和探针编写，不勾快照/缓存对照/最终定位。下一步上传探针，在同名 tmux/GPU0 新目录运行一次。

### 快照生成摘要回收

- 用户回传 PROBE_SUMMARY：快照 7781984 字节，SHA256 `c8682923be50a74c6884acc3e5bc1bdf5443ae33c364763092a9ceecbdd62b63`；29327 条事件，37 segment_alloc / 9764 alloc / 9763 free_requested / 9763 free_completed。截断标志 false，事件数低于 200000。
- decoder=21、边界记录=44，符合预期；摘要另存 `notes/week5/vae-history-01-user-summary.json`，明确为用户回传、不是原件本机核验。
- 当前未收到原 result/probe/snapshot/视频，带探针耗时和峰值尚未知。下一步 scp 传回全目录再核验哈希/身份/事件并分析峰值存活分配；不据事件数猜峰值来源或判泄漏，不重跑探针。

### history 原件验收及峰值归属分析完成

- 用户已传回整个 `history-baseline-01/`，本机 snapshot 哈希/大小/29327 条事件与摘要相同；input/result/probe、脚本、固定源 blob、输入及基线身份检查通过。history 已关闭，status=ok。
- MP4 完整解码 81 帧、832×480、16 fps、无全黑帧；与无探针及冻结 P01/s42 基线逐字节相同。探针记录为 10.147731060991646 秒、allocated 14698.970703125 MiB、reserved 18674 MiB，不用单次时间差推断加速。
- 新增 `analyze_vae_history.py`，按默认 native allocator 规则回放分配/释放/合并及块内余量，匹配所有最终快照块和 44 个边界记录。事件 28595 的 allocated 15412987904 字节与记录精确相同；requested 总量少 2861498 字节，不混用口径。
- 第 21 次 decoder 内 native `slow_conv_dilated_all_cuda_template<3>` 的 columns 临时缓冲区 requested 7897.5 MiB、block 7898 MiB，是峰值最大存活分配，栈为 vae.py:38/209/453/586；不是输出 cat 或开头 conv2。具体 backend 选择原因未测明。
- 新增 `attribute_vae_peak.py`，用前后观测的 storage allocation epoch 和 vae.py:205/209/210 的赋值顺序重建 `_feat_map` 成员 3556.083984375 MiB；另有尚未赋值 clone 292.5 MiB。临时形状明确标为源码/字节重建，没有内部直接 shape hook。
- 首次分析的绘图因独立依赖 cp312 与项目 Python 3.11 不匹配中止，部分目录 `vae-history-01-analysis/` 保留。纯分析与绘图分开后，最终目录为 `vae-history-01-analysis-v2/`；`plot_vae_history.py` 使用既有 bundled Python 3.12，静态曲线已查看，推理环境未变。
- PyTorch memory viz HTML 已生成；工具浏览器拒绝 file: 协议，未取得交互截图、未绕过策略。该截图和同输入 CPU 缓存对照继续待办，Week5 未收尾。当前只测量，不实施 Week6–7 改法。
- 计划、报告、预检、README、执行清单及 REPRO 已同步。下一步先在原 tmux 会话回传 GPU0 当前占用，再用原基础脚本增加 `--vae-cache-cpu`，新目录 `decode-cache-cpu-01` 测一次；本机 dry-run 参数核对通过，服务器未执行该对照。

### CPU 缓存对照摘要回收

- 用户确认无占用后执行 CPU 缓存对照，回传部分 stdout；摘要不包含 status。allocated=11683311616 字节（11142.07421875 MiB），reserved=16078 MiB，50.383249560996774 秒，输出 CPU FP32 `[3,81,480,832]`。
- 与已验收的无探针 baseline 本机复算：allocated 减少 3556.896484375 MiB / 24.198269%，reserved 减少 2596 MiB，单次耗时为 4.582868 倍。降幅与重建缓存差 0.8125 MiB，只作交叉验证线索，不据此认定相同峰值位置或唯一原因。
- 摘要另存 `notes/week5/vae-cache-cpu-01-user-summary.json`。原 input/result/video 尚未传回，本机未核实 cache 开关、状态、输入身份和视频一致性；计划对照项保持未勾选。下一步 scp 传回整目录，无需重跑；交互截图及 Week5 收尾继续待办。

### CPU 缓存原件验收完成

- 用户已传回三个原始文件。新增 `scripts/audit_vae_cache_cpu.py`，核实 input/result 交叉一致，最终 status=ok、cache_cpu=true、memory_history=false；与无探针 baseline 除保存路径外仅增加 `--vae_cache_cpu`，输入/模型/环境/源码身份一致，基础脚本和基线原件哈希未变。
- 本机完整解码 81 帧，832×480、16 fps、H.264/yuv420p，无全黑帧；MP4 SHA256 与无探针及冻结 P01/s42 基线相同，RGB 帧摘要也相同。没有原始浮点输出张量误差审计，不将视频相同外推全部任务无损。
- 审计输出为 `notes/week5/vae-cache-cpu-01-local-audit.json`，单次对照差值复算匹配用户摘要。allocated 减少 24.198269%，耗时为 4.582868 倍；本周测量不等于独立 GPU 重跑、正式速度验收或完整生成收益。
- CPU 缓存对照项已勾选，报告、权衡表、计划、README、执行清单、预检与 REPRO 同步。无需再跑 GPU 实验；仅余 memory viz 交互查看/截图与最终收尾。此前工具浏览器 file: 协议被拦截，计划下一步由用户本机普通浏览器打开已有 HTML 回传实际画面，不伪造完成状态。

### memory viz 截图回收与 Week5 收尾

- 用户回传截图，显示设备 0、Active Memory Timeline、9764/9764 分配、已渲染时间线及选中分配的栈。当前选中 292.5 MiB，不称截图选中了最大峰值块；最大块归属来自独立快照/源码审计。
- 原图复制到 `notes/week5/vae-history-01-analysis-v2/memory_viz-user-timeline.png`，305377 字节，复制前后 SHA256 均为 `4d4ac320b02198decd7131266ab1111e0aad921c162e23cc42814d0477d8e15c`。来源/边界另存 `memory-viz-user-screenshot.json`；这是用户浏览器截图，不冒称助手操作浏览器。
- Day7 和本周出口项按证据勾选完成，定位报告更新为 final_v4；README、执行清单、预检及 REPRO 同步，新建 `notes/REVIEW/WEEK5_REVIEW.md`。Week5 已收尾。
- Sage 综合结论不足、两种 FP8 不推荐、评分元数据/程度缺口、单输入/单次诊断、临时形状重建等边界保持。未实施 Week6–7 改法，未变更公开副本或发布。
