# Week4 日志

## 2026-10-26：Day 1 算子检查及本机验收

- 用户执行服务器命令，确认固定上游 commit，以及 bench、gen_eval、baseline 配置和冻结提示词的哈希与本机一致。
- 服务器初次检查缺少候选 sageattn / fp8 / fp8_scaled YAML；随后用户同步并贴回四份 eval YAML 哈希，本机重新计算后全部一致，Day 1 第一项文件核对完成。
- 使用副本 Python 绝对路径，确认 torch 2.9.1+cu128、Triton 3.5.1、SageAttention 2.2.0 及副本安装位置。
- 读取已安装 varlen 源码：Triton 分块 INT8 Q/K、FP16 V、FP32 softmax 状态和外层累加器，P×V 分块乘法输出指定 FP16。
- 新增 `scripts/check_sage_varlen.py`，本机语法、命令行和 CPU 误差计算检查通过；未在本机安装或执行 SageAttention。
- 用户在服务器 GPU 0 运行会话 `varlen-20261026T132946Z-a1c64053`，固定随机输入下两项检查输出均 finite。
- 自注意力 SDPA / Sage 均值为 40.665722 / 33.291090 ms，耗时下降约 18.13%；交叉注意力为 0.541247 / 1.047834 ms，耗时增加约 93.60%。
- 用户传回五份原始文件，本机独立核验脚本及三份源码哈希、计时条数、均值、范围和加速比，结果一致。未归档原始张量，不声称独立重算误差或输入哈希。
- 详情：`notes/week4_sage.md`；原件及本机审计侧记：`notes/week4/varlen-20261026T132946Z-a1c64053/`。
- 状态：Day 1 五项已验收。模型冒烟、trace、环境样本复现、正式性能和质量均未执行或验收。
- 本机另外按结构化 YAML 比较三个候选与 baseline，确认每个仅改变预登记的行为字段及 config_name / save_path；fp8_fast、compile、block swap 未启用。

下一步：进入 Day 2 的副本环境 GPU 0、2 步 stages 冒烟，预热 0、正式 1，检查四次前向及阶段校验。代码核对确认现有 bench.py 仍保存原始结果到 `notes/week2/`、latent/视频到 `outputs/week2/` 下的独立会话目录，沿用该行为并按终端实际路径归档；不修改 bench.py。

## 2026-10-26：Day 2 冒烟已收到服务器输出

- 用户在副本环境 GPU 0 执行 2 步 stages、预热 0、正式 1，会话为 `W4-sage-smoke-stages-20261026-214619-9754d61f`。
- 终端显示两步采样、latent 保存、VAE 解码、MP4 保存均结束，并写出 result.json 和 summary.json；显示耗时 34.142 秒、allocated 峰值 14698.4 MiB。
- 原始记录位于服务器 `notes/week2/<会话>/`，产物位于 `outputs/week2/<会话>/run-01/`。
- 原始文件已传回；本机复算四次前向及阶段计时，通过原脚本校验；summary 与重算完全一致。
- CPU 检查保存的 FP32 latent `[16,21,60,104]` 全部有限；视频完整解码为 81 帧、480×832、16 fps，无全零帧；0/40/80 帧像素均值约 90.50/90.42/90.86，方差约 783.17/770.63/775.81。
- 本机查看抽帧图：严重模糊、主体不清，但不是黑屏。两步样本不构成正式质量通过或失败归因，不推断 SageAttention 已造成退化。
- 用户随后贴回服务器 latent / MP4 哈希，本机重新计算并核对，两项全部一致；审计文件已保存到 `notes/week4/smoke-audit-W4-sage-smoke-stages-20261026-214619-9754d61f/`，Day 2 冒烟及 NaN/黑屏检查两项已验收。
- 本机已确认 CPU 读取 latent 和解码视频所需的 torch、safetensors、OpenCV、Pillow 可导入定位；没有安装依赖或启动本机 GPU 推理。

后续已收到下列 20 步 profile 的用户输出；冒烟无需重跑。

## 2026-10-26：Day 2 profile 已收到服务器输出

- 会话 `W4-sage-profile-profile-20261026-222504-af72f220`，副本环境 GPU 0，20 步、预热 0、正式 1，BF16 日志，FP8 未启用。
- 终端显示采样 20/20、latent 保存、VAE 解码和 MP4 保存结束，result.json 与 summary.json 已写出。
- 显示耗时 138.178 秒、allocated 峰值 14698.4 MiB；包含 profiler 开销，不用作速度门槛依据或与历史基线直接比较。
- 原始记录及产物已传回，本机确认 status ok、40 次前向、trace_ready=true，bench/配置哈希一致，summary 复算完全一致。
- 捕获 CPU/GPU Step#18–#21；旧 External-id 过滤会漏掉 Sage Triton 720 条内核。新增分析脚本以 CUDA launch correlation、同名 CPU/GPU Step 区间补回，最终包含 attention 240 次、Q/K 量化 480 次，未发现 PyTorch flash 前向内核；剩余排除的27条全部在捕获边界前。
- 同一规则重新分析历史 R5 与原结果一致。候选 attention 内核和量化分别为 4.236092 秒/44.41%、0.063730 秒/0.67%，只作机理诊断，不包含完整包装函数开销，不作为速度门槛依据。
- CPU 检查20步 latent 全有限；视频完整解码81帧，抽帧人物和举手可辨、非黑屏，不作质量通过结论。
- 分析脚本为 `scripts/analyze_week4_profile.py`，真实原始数据核查已通过；审计和抽帧位于 `notes/week4/profile-audit-W4-sage-profile-profile-20261026-222504-af72f220/`。
- 用户询问目录名称，已说明 bench.py 硬编码 week2 原始路径；用 W4 标签登记本周实验，新增分析放 week4，不移动原始文件、不修改已验收的 bench.py。
- 用户补充服务器 trace/latent/MP4 三项哈希，本机本轮重新计算并逐项比对，全部一致；Day 2 内核确认已验收，环境 P01/s42 复现未执行。

下一步：副本 Python 执行 gen_eval 的 torch 基线 P01/s42、20步、tag envcheck 单样本复现；本机 dry-run 确认只选一条，命名空间为 baseline-envcheck。输出位于 outputs/eval/baseline-envcheck，收据位于 notes/eval/baseline-envcheck。此单样本不要求全36条汇总；不一致则按计划停止，不自动重生成36段基线。
原基线 P01_s42 本机MP4已重新核对，哈希为 fa59fbd653bef48711826054cbfc3463da9003b10c429c2273d9cf77eb838217，与历史清单一致。单样本一致不等于完整环境或全部评测集输出等价。

## 2026-10-26：副本环境 P01/s42 复现已有服务器结果

- 用户提供 baseline-envcheck 输出，选中1条、20步，生成及保存结束，P01_s42: ok。
- 用户服务器 sha256sum 显示原 baseline 和新 baseline-envcheck MP4 均为 fa59fbd653bef48711826054cbfc3463da9003b10c429c2273d9cf77eb838217；本机本轮重新计算旧基线MP4并核对历史清单，结果一致。
- 这是固定样本的服务器复现证据，不证明全部36段或完整环境输出等价。本机没有重跑服务器任务，也未读取这次新视频。
- events 文件为 events-0-of-1-a35d671ac4ed4259817d9fbd6ba983ff.jsonl，原始收据在 notes/eval/baseline-envcheck/attempts/P01_s42/attempt-01/result.json，已传回并在本机核验。
- 收据确认 effective_args.attn_mode=torch；与旧收据相比生效参数仅保存路径不同，runtime 和负面提示词相同，identity/fingerprint 一致，四项身份哈希与本机当前文件相符；events 为 running→ok，只有一次尝试。
- 按预定的单样本MP4判据验收通过，审计为 notes/week4/envcheck_audit.json。旧/新latent文件哈希不同且新latent未传回，未比较张量值，不解释为张量一致或不一致，不更改预定判据，差异已留档。

Day2四项已验收。按既定安排沿用历史36段基线作为质量配对侧，说明环境检查仅覆盖P01/s42的MP4；正式性能仍需当期同环境基线。
随后用户执行下列Day3 A1；不以环境复现或历史均值替代A1。

## 2026-10-26：Day3 A1 当期 torch total 已收到服务器输出

- 会话 `W4-A1-torch-total-20261026-232448-8df71ea1`，副本 Python、GPU0、eval_baseline、total、sample-smi、20步、预热1+正式3。
- 终端显示 warmup-01 131.068 秒；run-01/02/03 分别 134.493/135.827/137.391 秒；四轮均完成生成及保存、result 写出，summary 已保存。各轮allocated终端摘要均为14698.4 MiB。
- 原始session/result/summary及四份gpu.csv已传回。本机确认全部status=ok、torch后端和冻结参数、正确的预热1+正式3、相符的commit/bench/config哈希和与Day2一致的环境字段。
- 正式耗时134.4933614092879、135.82714506983757、137.39130416186526秒，均值135.90393688033024秒，范围134.4933614092879–137.39130416186526秒。用原始四轮result重算summary完全一致，替代终端舍入值暂算结果。
- 三次正式allocated/reserved/整卡采样峰值分别为14698.353515625/18666/19627 MiB；预热同样。CSV行数warmup/run01/run02/run03为1309/1342/1356/1372，所有行均可解析，索引/时间戳/显存/利用率及峰值与JSON一致，采样warning均为null。
- A1原始记录验收通过；审计及原始文件哈希在notes/week4/A1_audit.json，性能表在notes/week4_perf_attention.md。未传回计时视频/latent，不作质量验收；没有独立验证全程GPU1或进程占用。
- A2/A3/A4未执行，不提前给出优化收益。

下一步：A2独立新进程，同副本环境、GPU0、eval_sageattn、total + sample-smi、预热1+正式3、标签W4-A2-sage，GPU1保持不并行生成。

## 2026-10-27：A2 已保存结果，原始文件待传回

- 用户关闭终端后读取 session/summary，会话为 W4-A2-sage-total-20261026-235430-c79c6260，session.status=ok，正式3次，不需重跑。
- 用户提供summary均值125.05904630338773秒，范围123.91872290102765–125.74216801207513秒；allocated/reserved为14698.353515625/18666 MiB，整卡采样峰值待CSV读取。
- 与已验收A1相比暂算耗时下降7.97982076596643%，双方范围不重叠；如原始数据核验一致，达到预登记性能门槛且不触发追加轮次。A2原始逐次JSON/CSV尚未传回，不提前作最终性能或质量结论。
- session展示的配置、commit、bench/config哈希、torch/CUDA/GPU等记录与预期相符。

下一步：传回A2完整notes记录目录，核验全部result/CSV及summary，再进行A3/A4。Day3总项仍未完成。

## 2026-10-27：A2原始记录验收通过

- 本机核验A2四轮result、session、summary及CSV，全部status=ok、预热1+正式3，冻结设置和sageattn后端正确，记录环境与A1一致；配置行为变化仅attn_mode。
- 三次正式123.91872290102765、125.74216801207513、125.51624799706042秒，预热122.7151336800307秒；原bench.make_summary重算与传回summary完全一致。
- 四份CSV行数1226/1237/1255/1253，索引、时间戳、显存、利用率及采样峰值核验通过，warning均null。
- A2正式均值125.05904630338773秒，相对A1耗时下降7.97982076596643%，范围不重叠，按预登记速度门槛通过，不触发A5/A6；不作为质量通过或最终可用结论。
- allocated/reserved/整卡采样峰值各为14698.353515625/18666/19627 MiB，与A1相同。未观察到整次峰值下降，阶段证据待A3/A4。
- 审计及原始文件哈希为notes/week4/A2_audit.json，性能表已同步。

下一步A3：eval_baseline、stages、20步、预热1+正式1、副本环境GPU0，GPU1不并行生成。之后新进程运行A4，阶段值只用于机理，不改动A1/A2速度结论。Day3总项仍未完成。

## 2026-10-27：A3收到用户服务器输出

- 会话 `W4-A3-torch-stages-stages-20261027-144838-89be411c`，stages、20步、预热1+正式1。
- 终端显示两轮均完成采样和latent/视频保存、result写出、summary写出；预热128.611秒，正式134.512秒，allocated摘要均14698.4 MiB。
- 原始JSON尚未传回，40次前向、阶段调用、阶段总和和正式原始精度未核验，A3暂不标为验收通过。A4未执行，Day3总项保持未完成。
- 阶段测量只作机理解释，不用此次134.512秒替换A1或与A2 total比较收益。

下一步：用户传回A3 notes记录目录，本机验收后再执行A4，无需重跑A3。

## 2026-10-27：A3原始记录验收通过

- session、summary、预热和正式result已传回。本机核验status、冻结设置、torch后端、bench/config哈希及与A1一致的记录环境字段。
- 预热及正式均40次前向、八项阶段函数各一次，阶段区间聚合正确，原bench.make_summary复算与summary完全一致。
- 正式总耗时134.51247651409358秒，阶段总和134.51209623692557秒，差0.0002827077293204554%；预热128.61072522820905秒不纳入正式统计。
- 去噪111.18730131816119秒，占stages总耗时82.65947085325688%；T5加载10.60615798085928秒，VAE解码（不含loader）10.317400565836579秒。
- allocated峰值位于VAE解码14698.353515625 MiB；去噪allocated/reserved峰值4183.47216796875/5974 MiB。本次无整卡采样，不混用total的显存采样。
- 正式每前向均值2.7714400857919825秒，相邻两次前向均值5.542880171583965秒，不等同完整采样步，也不把40次前向当独立生成样本。
- 审计为notes/week4/A3_audit.json，阶段表已写入notes/week4_perf_attention.md。A4未执行，Day3总项仍未完成。

下一步：A4副本环境GPU0、eval_sageattn、stages、20步、预热1+正式1、标签W4-A4-sage-stages；GPU1不并行生成，阶段对照仅用于解释机理。

## 2026-10-27：A4收到用户服务器输出

- 会话 `W4-A4-sage-stages-stages-20261027-150717-7519d3ee`，副本Python、GPU0、eval_sageattn、stages、20步、预热1+正式1。
- 终端显示两轮采样和latent/视频保存结束，result及summary已写出。舍入值为预热119.041秒、正式122.616秒，allocated摘要均14698.4 MiB。
- 原始JSON尚未传回，status/实际参数、40次前向、阶段聚合及summary待本机核验；A4暂不标为验收通过。
- 不用stages总耗时替换A2 total，不将A3/A4总差直接归因于attention；Day3总项保持未完成。

下一步：传回A4 notes记录目录，核验并整理A3/A4阶段表和Day1算子诊断的关系，无需重跑测量。

## 2026-10-27：A4验收及Day3收尾

- A4原始session/result/summary已本机核验，全部status=ok，副本记录环境与A3一致，配置行为仅切换attn_mode；两轮各40次前向、八阶段各调用一次，聚合、峰值和summary复算一致。
- 正式总耗时122.6158517957665秒，去噪100.19834579480812秒；相对A3去噪少10.98895552335307秒、9.88328%，占stages总差值11.896624718327075秒的92.3704%。其他阶段仍有波动，不能将全部差值作attention因果收益。
- A4每前向均值2.4947854763478974秒，相邻两次和均值4.989570952695795秒；峰值仍位于VAE解码，整次峰值未下降。
- Day1按每层self+cross各一次粗算算子下降16.6672%，结合历史attention kernel忙碌占比53.8986%近似估算去噪下降8.9834%、端到端7.4256%；与实测去噪9.8833%和total7.9798%只作量级对照，明确随机输入/历史kernel与墙钟口径不等价等假设。
- 审计为notes/week4/A4_audit.json；notes/week4_perf_attention.md已补全total、stages、三种显存、前向及机理表。保留原预期，未用stages替换total门槛结果。
- Day3出口满足；A5/A6未触发，不执行。质量/36段可靠性未评估，不给最终可用分类。
- 本机只读检查Day4计划：sageattn分片0/2、1/2各18条，互不重叠、并集36，20步、sageattn命名空间，无模型导入或生成。服务器GPU1 Sage路径仍待实际执行，不外推已通过。

下一步：Day4评测生成准备，先核对服务器两卡任务占用及tmux可用，再按两卡分片推进。尚未启动服务器36段生成。

## 2026-10-27：Day4服务器汇总36/36成功，本机验收待传回

- 用户报告两卡无其他任务、tmux 3.4；先启动GPU0分片0/2，日志确认P01_s42: ok。
- GPU1先以分片1/2、limit 1生成P01_s1001，用户报告ok；随后同分片去掉limit继续，既有成功样本由脚本校验并跳过，不额外增加样本。
- 用户提供两边末尾Worker log，GPU0最后样本P12_s1001: ok，GPU1最后样本P12_s2002: ok。
- 用户运行同配置的--summarize，返回Coverage: {'ok': 36}，清单为服务器notes/eval/sageattn.jsonl。按已核对的脚本语义，汇总会检查最终收据及视频/latent文件哈希；这是用户服务器输出，本机尚未读取本次36段原件。
- 不由最终成功数推断从未失败；首次失败、尝试次数、GPU分配及冻结参数需传回原始收据/events后核查。
- 下一步传回notes/eval/sageattn目录、sageattn.jsonl和稳定命名的36段MP4，不传latent或outputs中的.attempts。本机接收目录已建立；本机视频哈希验收和指标尚未执行，不标记Day4整体完成。

## 2026-10-27：Day4生成本机验收通过，指标计算启动

- 已读取完整36条清单、dataset/coverage、36份尝试收据及3份events，并独立核算36段本机MP4哈希，全部一致；样本集合与冻结提示词×种子完全匹配，GPU0/1各18条。
- 尝试记录均attempt-01、ok，无失败和重试记录；events为36 running、36 ok、1 skipped（GPU1首条P01_s1001续跑时验证后跳过）。无残留样本锁。
- 本机审计为notes/week4/generation_audit.json。latent按计划未传回，不声称独立核验其内容或哈希；服务器summarize已检查。生成成功不代表质量通过。
- 现有metrics.py固定读取第三周基线与探针；新增scripts/metrics_candidate.py复用原计算函数，未修改原脚本、协议、旧指标文件或输入视频。
- dry-run确认候选和基线36对、两侧视频哈希、协议/函数脚本哈希及依赖版本一致；CPU FP32、固定CLIP revision、80个SSIM帧对和21个CLIP采样帧沿用第三周。
- 本机指标计算已启动，输出notes/week4_metrics_sageattn，逐条保存结果、四个指标差值及全部差值绝对值排名，不新增质量阈值。结果完成和独立复算前，Day4最后一项不勾选。

## 2026-10-27：Day4指标完成及收尾

- scripts/metrics_candidate.py实际计算36/36成功、0error，CPU FP32，离线耗时150.414秒，2880个SSIM帧对值和756个CLIP帧值。原解码函数强制校验81帧、480×832、16fps。
- 本机逐项复算每条四标量、36对baseline/candidate差值及全部绝对差值排序，结果一致；核验运行元数据和输入/参考/协议/函数及入口哈希。没有第二次重复运行指标模型。
- 输出notes/week4_metrics_sageattn；复算审计notes/week4/metrics_sageattn_audit.json；说明及差值较大样本记录notes/week4_metrics_sageattn.md。
- 原metrics.py哈希、第三周run/per_video输出保持不变，不重算基线、不改视频、不安装依赖。指标只作辅助线索，无新增质量阈值或样本剔除，不提供给匿名评分者。
- Day4四项已验收。36/36可靠性覆盖和性能门槛已取得，人工质量门槛未评估，不能给最终可用结论。

下一步Day5：准备36对左右随机配对及固定12段Sage单视频匿名包，私人映射单独保存，沿用R01/判据v1.0，不要求人工耗时。该包尚未制作或发出。

## 2026-10-27：Day5匿名评分包制作完成，未发送

- 新增blind_pack_candidate.py，复用旧编码/动作要求，36配对+冻结12单视频随机混排为T001–T048，左右独立随机；未重抽子集或更改判据。
- 工作簿用spreadsheets技能的Artifact Tool制作，两页评分记录/会话记录，单视频和配对分别下拉、所有评分空白，不含人工耗时栏；用加载器的bundled Node及依赖，通过公式响应、空白恢复、XML校验和各页渲染。
- 48段播放副本统一CRF18重编码，原始视频未修改。verify_candidate_videos.py重放随机化、校验全部来源/方向/哈希/格式及0/35/80帧内容。
- ZIP为outputs/week4-R01-20261027.zip，53个成员、77319728字节，CRC及文件哈希核验通过；SHA256为b4c1436fcdef60750d6679c397fdb0a436d45174bdf67369299e46af830431fb。
- 私人映射、验证结果及预览在notes/blind_private/week4-R01-20261027，未混入公开目录、工作簿或ZIP。未显示配置/速度/自动指标给评分者。
- 说明为notes/week4_blind.md。当前只完成包制作；未通过工具发送任何消息或文件，尚无评分回收。R01既往接触需在实际会话记录，不宣称严格独立盲评。

下一步：用户将公开ZIP交给R01，按包内说明完成48项评分，返回原始XLSX；收回后先归档及哈希，再解盲和澄清冲突，不自动修改原答。

## 2026-10-27：评分回收，解盲前核对

- 用户返回评分表，48项均已填写；源文件及归档SHA256一致，为bf3a3a2dd254b00f21b880edf27a757e77e42673bd0384549a7653f3e4ed1a40，原件未修改。
- 新增只读audit_candidate_ratings.py，核对公开任务身份并保存pre_unblind_audit.json；本轮未打开私人映射。
- T001/T002/T003/T010符合度、T032及T038三维度存在方向/程度冲突，共6个编号10项，待评分者澄清，不自动改成无差异或轻微退化。
- 单视频三维度分别为10/2、10/2、9/3项的2分/1分，无0分；仅作绝对质量分布，不能替代相对质量判断。
- 用户报告配对基本接近，人物抽动和蒸汽物理有缺陷；作为补充观察保留，不自动归因于Sage。播放/观看方式、问题秒数及会话元数据缺失仍保留。
- 详情notes/week4_rating_review.md。回收与归档完成，冲突澄清、解盲和最终质量结论仍待完成。

## 2026-10-27：SageAttention质量本轮评估完成，结论不足

- 两次补充原话已独立归档。用户确认T032的B、T038的A动作更合理，但只略稳定；据此仅澄清两项时序为轻微，未扩展为其他维度改善。
- 按用户本次开展评估的要求，先保存quality_decisions_before_unblinding.json及effective_answers_before_unblinding.json，再读取映射；不等全部冲突解决才统计，但保留未解决项并禁止判通过。
- 新增analyze_candidate_ratings.py，只读复核原表与审计、48项身份、36对和固定12段子集，重新核对72段原始基线/Sage视频哈希。原表和归档哈希不变；未改动视频、规则或旧分析入口。
- 结构完整34项、无明显差异34、未解决2；时序完整36、无明显差异34、轻微更好1、轻微更差1、明显变差0；符合度完整30、无明显差异29、有差异难判优劣1、未解决6。完整条目不等于已验证完整原速观看。
- 符合度4项的“轻微”是绝对提示词偏离，原无明确优劣方向保留，但不自动补相对程度子类；另2项与结构2项仍有原答冲突。未解决不混入评分者“无法判断”。
- T019原文实际为“整体可以，胳膊扭曲诡异”，原分2/2/1保留；更正先前摘要转述，不将其推断为新增明显失败。
- 逐项JSON/CSV及输入哈希保存到原件归档目录；CSV计数另用PowerShell分组复核。质量报告notes/week4_quality_attention.md，结论卡notes/week4_result_attention.md。
- 已确认速度下降7.9798%、36/36生成成功、整次显存不降；时序计数达标，但未解决评分和观看/会话信息缺失按预登记优先级使最终分类为结论不足。未确认新增明显退化，不写成质量失败或已通过。
- 本次未重跑推理/性能/指标、未代替用户重新观看或改分。未来澄清须附记助手已解盲，不回写本轮统计或冒称新严格盲评。FP8尚未启动。

## 2026-10-27：普通FP8冒烟收到服务器日志

- 会话W4-fp8-smoke-stages-20261027-233635-3ec2be40，stages、2步、无预热、正式1次、cuda:0。
- 日志显示DiT BF16计算精度、float8_e4m3fn权重精度、VAE BF16，fp8_scaled=false，全部权重键匹配，明确执行FP8权重cast。
- 两步采样、latent保存、VAE解码及MP4保存完成，打印result/summary；终端值33.127秒、allocated峰值14698.4 MiB，仅为两步冒烟舍入值。
- 主环境SageAttention可选导入失败不影响本次torch attention配置，不为本组安装Sage或切换副本环境。
- 原始JSON、产物和哈希尚未传回，fp8_fast生效值、阶段检查、latent有限值与视频内容待核验；不据日志宣布FP8矩阵乘加速或质量通过。
- 新增notes/week4_fp8.md。scaled冒烟、源码核对及正式性能仍待进行，计划复合冒烟项不勾选。

## 2026-10-27：Scaled FP8冒烟收到服务器日志

- 会话W4-fp8-scaled-smoke-stages-20261027-234012-aee81310，stages、2步、无预热、正式1次、cuda:0。
- fp8_scaled=true，加载使用FP8 optimization；日志记录300个Linear层优化及300个前向替换，全部权重键匹配。
- 两步采样、latent及MP4保存完成，终端33.541秒、allocated峰值14698.4 MiB；单次两步数值不构成正式速度对照。
- 静态检查本机保存源码：scaled设置dit_weight_dtype=None是有意保留不同权重类型；args.fp8_fast控制use_scaled_mm，false分支反量化后用输入类型F.linear，不能凭层数日志宣称硬件FP8矩阵乘。
- 两次原始session/result/summary及latent/视频尚未传回，实际fp8_fast、finite及内容核验待完成；原计划冒烟和完整源码项仍不勾选。
- 下一步打包传回两个notes会话目录及两个outputs会话目录，按原路径归档后本机验收，不重跑或提前进入正式计时。

## 2026-10-28：两种FP8冒烟本机验收完成

- 用户解压目录为notes/week4/W4-fp8-smoke-artifacts-20261027，10份原件齐全，与本机tar成员逐项哈希一致。tar本机SHA256为fc760f908e46102ce12a537ea95d67d164b06fd820c244e27882307d65170372；未提供服务器tar哈希，不声称跨机器独立对照。
- 新增scripts/audit_fp8_smoke.py，用已有项目环境CPU读取，不运行生成、不安装依赖、不修改输入。核对session/result状态、生效配置、bench/config哈希、commit、相同runtime、4次前向、阶段聚合和summary，全部通过。
- 两份float32 latent均[16,21,60,104]且全finite；两段H264完整解码81帧、832×480、16fps，均非全黑。第0/40/80帧预览已本机查看，明显模糊，只通过冒烟有效性，不通过画质。
- 普通/scaled原始耗时33.126806498039514/33.54078897414729秒，仅为单次两步诊断。allocated整次峰值均14698.353515625 MiB、reserved均18666 MiB，allocated峰值位于VAE解码；本次无整卡采样。
- 三份保存源码与本机固定commit f8a1b03794a49239a3539015075f5123d6c07d66对象一致（仅允许换行差异）。普通FP8权重cast后标准Linear/BF16 autocast；scaled生效fp8_fast=false，反量化后F.linear，不走专门_scaled_mm分支。不外推每个算子均BF16或GPU指令级验证。
- 审计notes/week4/fp8_smoke_audit.json，预览fp8_smoke_preview.png，文档notes/week4_fp8.md。Day6两个FP8准备项已完成。
- 下一步服务器主环境GPU0运行当期baseline total，预热1+正式3、sample-smi，GPU1不并行生成；随后再测fp8和scaled，不复用副本A1。正式性能尚未执行。

## 2026-10-28：F0主环境基线收到服务器摘要

- 会话W4-F0-torch-total-20261028-003533-43e7c074，用户提供session/summary：status=ok，total、20步、预热1+正式3、GPU0、sample-smi。
- 正式均值134.03493864291036秒，范围133.4579865988344–135.17331278230995秒；三次allocated峰值均14698.353515625 MiB，reserved均18666 MiB。
- 记录配置为torch、各FP8开关关闭、固定提示词/种子及生成参数；配置/bench哈希和commit与冻结记录相符。
- 仅服务器粘贴输出；原始逐轮result及gpu.csv未传回，本机尚未复算摘要或整卡采样峰值。不混用副本A1。
- 详见notes/week4_fp8.md；下一步F1普通FP8 total，同环境GPU0、预热1+正式3，继续在w4-fp8-bench tmux中运行。

## 2026-10-28：F1普通FP8首轮摘要，速度收益未确认

- 用户返回W4-F1-fp8-total-20261028-133106-bf2ccdec的run-03末尾及summary，正式3次mean132.43742388285077秒，min/max131.07531786011532–133.9174454039894秒。
- 相对F0摘要，均值少1.597514760059596秒、下降1.1918644319416003%；未到5%，且范围交集为133.4579865988344–133.9174454039894秒。
- 三次allocated均14698.353515625 MiB、reserved均18666 MiB，与F0摘要相同。整卡峰值及逐轮原始记录待传回验收，不给最终资源/质量分类。
- 冻结规则触发唯一一轮baseline与普通FP8双方追加，各预热1+正式3，分别合并全部6次。初始范围已重叠且合并保留极值，不能期待追加消除已有重叠或通过反复测量追求通过。
- 先完成已安排F2 scaled首轮，不并行追加计时；随后再安排确认轮次和stages，暂不开始72段生成。

## 2026-10-28：F2 scaled摘要与确认轮安排

- 会话W4-F2-fp8-scaled-total-20261028-135518-9da4d16a，用户返回run-03末尾和summary主要字段：3次mean134.91440854842463秒，范围134.57658117823303–135.53854936733842秒。粘贴末尾不完整，原JSON待传回。
- 比F0均值多0.8794699055142701秒、0.6561497430586459%，范围交集134.57658117823303–135.17331278230995秒；首轮不确认提速，也不据此认定稳定变慢。
- allocated/reserved峰值仍14698.353515625/18666 MiB，整卡和逐轮聚合待原件验收。未推断DiT阶段无收益。
- 两候选均触发冻结的唯一确认轮。依据同一未变环境共享基线条款，计划顺序F3 torch-confirm、F4 fp8-confirm、F5 fp8-scaled-confirm，各预热1+正式3、GPU0及sample-smi，不并行。
- 最终F0+F3、F1+F4、F2+F5各合并6次，再作最终时间/显存判断；本次只给F3命令，确认轮均尚未验证执行。

## 2026-10-28：F3服务器摘要完成，继续F4

- 会话W4-F3-torch-confirm-total-20261028-142252-ac3ea804，用户提供status=ok、20步total、预热1+正式3、GPU0、sample-smi及冻结基线配置；记录环境/哈希与F0相符。
- 正式mean134.02658165665343秒，范围133.9448650549166–134.0878832731396秒，allocated/reserved各次14698.353515625/18666 MiB。
- F0+F3摘要合并六次mean约134.0307601497819秒，范围仍133.4579865988344–135.17331278230995秒；逐轮原始数据与整卡CSV待传回核验。
- 下一步F4普通FP8确认轮，同环境20步total、预热1+正式3、GPU0/sample-smi；随后F5，尚不出最终收益分类。

## 2026-10-28：F4摘要完成，普通FP8六次速度门槛未达

- 用户返回W4-F4-fp8-confirm-total-20261028-144843-947405db session/summary，status=ok、正式3次，参数/哈希/commit和记录环境与预定普通FP8路径相符。
- F4 mean132.23342725541443秒，范围132.08548581693321–132.50717843603343秒；allocated/reserved各次14698.353515625/18666 MiB。
- F1+F4六次摘要合并mean132.3354255691326秒、范围131.07531786011532–133.9174454039894秒；相对F0+F3少1.2648847016570715%，低于5%，合并范围仍重叠。
- 据现有服务器摘要普通FP8速度门槛未达；保留全部六次，不再追加total。原始逐轮及整卡CSV待本机核验，阶段显存/质量结论待后续。
- 下一步F5 scaled唯一确认轮，GPU0、主环境、20步total、预热1+正式3、sample-smi，不与其他生成并行。

## 2026-10-28：F5完成服务器摘要，total唯一确认轮结束

- 用户提供W4-F5-fp8-scaled-confirm-total-20261028-153815-14aeaba4 session/summary，status=ok、正式3次、20步、GPU0/sample-smi，scaled开关、哈希及runtime相符。
- F5 mean134.3752808373732秒，范围133.48830372607335–135.04951727204025秒；allocated/reserved各次14698.353515625/18666 MiB。
- F2+F5六次mean134.64484469289891秒、范围133.48830372607335–135.53854936733842秒；相对六次基线多0.4581668733586078%。
- 三组各六次的摘要汇总见notes/week4_fp8.md。普通FP8少1.2649%、scaled多0.4582%，均未达速度门槛且范围重叠，不再追加total。
- 以上仍为用户服务器摘要。下一步六个notes会话目录打包传回，本机核验逐轮result、summary、gpu.csv及哈希后再开展三个20步stages；不传total视频/latent，不并行生成72段。

## 2026-10-28：F0–F5 total原始记录本机验收通过

- 用户解压至notes/week4/W4-fp8-total-records-20261028，60份原始文件齐全，本机tar全部成员与解压文件哈希一致；tar SHA256为89b7b7fdc2c012d87a5379f234a97ed697560e6504b979f211656c27d3c50f97，服务器tar哈希未提供。
- 新增scripts/audit_fp8_total.py，6个session/24个result（含预热）均ok；配置/脚本哈希、commit、全部生效参数与记录runtime核对通过，total无阶段/profiler插桩，6份summary由未修改bench.make_summary精确复算一致。
- 24份CSV共31,692行，索引/时间序列/数值合法，行数和峰值与result一致、warning均null；输入哈希前后不变。
- 原始六次mean基线134.0307601497819、普通FP8 132.33542556913258、scaled 134.64484469289891秒，分别少1.2648847016571135%/多0.4581668733585865%，范围均重叠、速度门槛未达、不再追加total。
- 三组每次正式allocated/reserved/整卡采样峰值均为14698.353515625/18666/19627 MiB；整次无下降，不推断DiT阶段无收益。
- 审计notes/week4/fp8_total_audit.json，详情notes/week4_fp8.md。total视频/latent未传回，不做20步质量验收。
- 下一步F6基线stages→F7普通FP8 stages→F8 scaled stages，各20步、预热1+正式1、主环境GPU0且GPU1不并行。当前只安排F6，未收到stages执行证据。

## 2026-10-29：F6基线阶段测量收到完整日志

- 会话W4-F6-torch-stages-stages-20261029-151302-78d70e8b，stages、20步、预热1+正式1，两轮采样与latent/MP4保存均结束，打印最终summary。
- 终端预热129.010秒、正式135.730秒，allocated峰值均14698.4 MiB。仅日志舍入值，原始阶段聚合及40次前向待传回核验。
- 前次查找命令粘贴损坏不构成任务失败；现有日志确认流程完成，无需重跑。两步或阶段结果均不替换total门槛。
- 下一步F7普通FP8 stages，20步、预热1+正式1，主环境GPU0且GPU1不并行；随后F8。

## 2026-10-29：F7普通FP8阶段测量收到完整日志

- 会话W4-F7-fp8-stages-stages-20261029-152149-75a61563，20步stages、预热1+正式1，日志确认float8_e4m3fn权重、BF16 DiT/VAE及fp8_scaled=false，全部键匹配。
- 两轮采样和latent/MP4保存均结束，打印result及summary；预热129.493秒、正式133.552秒，allocated峰值均14698.4 MiB，仅日志舍入值。
- 原始参数、40次前向、阶段聚合及峰值待传回核验，不更新已验收total速度结论或推断DiT阶段收益。
- 下一步F8 scaled FP8 stages，主环境GPU0、20步、预热1+正式1，GPU1不并行。F8尚未执行验收。

## 2026-10-29：F8日志完成，三组阶段记录待传回

- 用户提供W4-F8-fp8-scaled-stages-stages-20261029-152931-00dfa303完整日志，20步、预热1+正式1；两轮scaled路径均有300层优化/前向替换，全部键匹配，产物及summary已保存。
- 预热132.960秒、正式136.515秒，allocated峰值均14698.4 MiB，均为日志舍入值，未当作阶段聚合验收。
- F6/F7/F8均已从服务器日志确认跑完，无需重跑。下一步传回三个notes会话原件，不需要视频/latent；本机检查参数、40次前向、阶段聚合/峰值及summary后再整理性能表。
- 不改变六次total速度门槛结论，72段评测尚未启动。

## 2026-10-29：F6–F8阶段原件验收及性能收尾

- 用户指向旧total目录；本机实际找到新stages tar，未见解压阶段文件。新增audit_fp8_stages.py验证精确成员集合/路径/类型后解压至独立stages目录，旧total记录不动。
- 12份JSON核对通过，6轮均ok，各40次前向、8个阶段各调用1次；配置/bench/commit、生效参数和total runtime一致，阶段聚合与三份summary精确复算一致。
- 本机tar SHA256 8afe8ff030a6773de366e16ad13d5d20625635fd9cbb420f393b9fadb3487882，成员与解压文件一致，未获服务器哈希。审计notes/week4/fp8_stages_audit.json。
- 基线/普通/scaled去噪allocated为4183.472168/2882.593750/2917.081543 MiB，候选下降31.0957%/30.2713%。
- DiT加载allocated为2732.646973/4139.415527/1553.215820 MiB：普通反而高51.48%，scaled低43.16%。类型转换存储重叠仅是可能解释，未采分配事件不作确证。
- 三组allocated整次峰值均在VAE解码14698.353516 MiB，reserved18666 MiB；已验收total整卡19627 MiB，无整次峰值下降。本次stages无整卡采样。
- 完整阶段耗时/三种内存表、机理和局限已写notes/week4_fp8.md。正式速度结论仍为两者均未达5%门槛，不再测total；每侧1次stages不替代total。
- 性能复合项已完成，72段生成、质量及最终分类未完成，不因阶段内存收益提前称可用。

## 2026-10-29：用户同意启动FP8两配置72段评测

- 本机只读核对gen_eval正式计划，两配置各18+18分片互斥且并集36，20步、torch、fp8_fast=false；脚本哈希保持冻结值。
- 顺序普通FP8 GPU0先启动、确认首条ok后GPU1；36段汇总后再scaled同样执行。使用主环境、正式命名空间、tmux及项目日志，保留成功跳过/失败/重试证据。
- 本次仅准备并给GPU0启动命令；服务器占用与预检输出、首条成功尚待用户返回，不标记72段生成完成。

## 2026-10-29：FP8两配置72段原件本机验收通过

- 用户按顺序完成两配置双卡生成及summarize，均36/36 ok；本机收到154文件tar，服务器/本机SHA256一致为4412ce239e719935db16148916d33d21b6914ae671ad930a4231d61aa7722869。
- 新增audit_fp8_generation.py，校验tar路径/类型及已有文件一致性后按原项目路径解压。72段MP4与manifest/收据哈希全部一致，逐段完整解码81帧、832×480、16fps且非黑屏。
- 两配置分别36份attempt-01 ok，GPU0/1各18，2个events文件各合计36 running/36 ok；无失败、重试、跳过或残留锁。冻结参数/identity/分片及记录runtime逐项核对正确。
- 审计notes/week4/fp8_generation_audit.json，说明notes/week4_fp8.md。latent按计划未传回，本机不声称独立核验latent；服务器summarize有哈希检查。
- Week4 FP8冒烟/源码/性能/72段生成均已完成。指标及人工质量仍属后续Week5工作，未计算或代评，不提前给最终可用结论；Sage未解决评分限制继续保留。
