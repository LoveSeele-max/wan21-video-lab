# Week4 FP8 对照

记录日期：2026-10-27；本机验收更新：2026-10-28，Asia/Shanghai。

## 冻结范围与顺序

主环境为服务器 `/home/user/projects/wan21-video/env`。普通FP8与scaled FP8分别测试；
两者均使用torch attention、fp8_fast=false，不与SageAttention叠加。
先各跑2步stages冒烟并核验，再核对固定源码实际前向路径。
正式性能需在主环境重新测baseline total、fp8 total、fp8_scaled total，各预热1+正式3，随后各测stages。
不能复用副本环境A1作本组速度对照，不用两步结果判定正式速度或质量。

## 普通FP8：收到服务器终端输出

会话：`W4-fp8-smoke-stages-20261027-233635-3ec2be40`。

用户返回的日志显示：

- stages、actual steps=2、warmup=0、measured=1，使用cuda:0。
- DiT precision=torch.bfloat16，weight precision=torch.float8_e4m3fn，VAE precision=torch.bfloat16。
- 模型创建时fp8_scaled=false；源权重BF16，加载匹配全部键后将模型转为float8_e4m3fn。
- “Loading state dict without FP8 optimization”之后有明确的FP8权重cast，不据前一句认定普通FP8未启用。
- 两步采样、保存latent、VAE解码及保存MP4完成，打印result与summary路径。
- 终端舍入值：33.127秒，allocated峰值14698.4 MiB；无整卡采样结果，不推断正式速度或省显存收益。
- SageAttention导入失败，但本组冻结配置使用torch attention，且本次完整流程继续跑完；不据此要求主环境安装SageAttention。

服务器记录目录：
`/home/user/projects/wan21-video/notes/week2/W4-fp8-smoke-stages-20261027-233635-3ec2be40/`。

服务器产物目录：
`/home/user/projects/wan21-video/outputs/week2/W4-fp8-smoke-stages-20261027-233635-3ec2be40/run-01/`。

latent：`20261027-233659-278_42_latent.safetensors`。
MP4：`20261027-233709-080_42_.mp4`。

以上为用户提供的服务器日志证据，本机尚未收到session/result/summary、latent或视频。
status、effective_args中的fp8_fast/attn_mode、四次前向、阶段检查、原始精度耗时及峰值待原JSON核验。
latent有限值、视频可解码/非黑屏与产物哈希也未验收，不将保存成功等同数值或画质通过。
主环境解释器、两卡占用、配置及脚本哈希的预检查输出尚未返回，不补造已确认状态。
仅由权重dtype日志不能确认FP8矩阵乘，实际前向反量化/运算路径待源码核对。

## Scaled FP8：收到服务器终端输出

会话：`W4-fp8-scaled-smoke-stages-20261027-234012-aee81310`。
用户提供的日志显示stages、2步、无预热、正式1次、cuda:0；DiT与VAE precision均为BF16，
weight precision=None，fp8_scaled=true，加载使用FP8 optimization。
日志记录300个optimized Linear layers、300个monkey-patched Linear layers，全部权重键匹配。
两步采样、latent保存、VAE解码及MP4保存完成，打印result/summary路径。
终端舍入值33.541秒、allocated峰值14698.4 MiB，不与普通FP8的单次两步耗时判定快慢。

服务器记录目录：
`/home/user/projects/wan21-video/notes/week2/W4-fp8-scaled-smoke-stages-20261027-234012-aee81310/`。
服务器产物目录：
`/home/user/projects/wan21-video/outputs/week2/W4-fp8-scaled-smoke-stages-20261027-234012-aee81310/run-01/`。
latent：`20261027-234036-134_42_latent.safetensors`；
MP4：`20261027-234045-950_42_.mp4`。

本次同样只有用户服务器日志，原始JSON、latent和视频待传回；不能确认有限值或画质。
`fp8_fast=false`是冻结配置值，实际effective_args仍待原JSON核对。

## Scaled路径的初步源码解释

已读取本机保存的 `notes/source/wan_generate_video.py`、`wan_model.py`、
`fp8_optimization_utils.py`；以下是该源码的静态解释，不替代服务器生效参数核验：

- get_generation_settings在fp8_scaled分支设置dit_weight_dtype=None，注释为不同权重精度、不统一转换类型；
  普通fp8分支才设置统一float8_e4m3fn。因此scaled日志的None本身是预期配置行为，不是未启用FP8。
- load_dit_model传递use_scaled_mm=args.fp8_fast，scaled加载会应用带scale_weight的Linear前向替换。
- fp8_linear_forward_patch的use_scaled_mm=false分支将权重转换到缩放因子类型并乘scale，
  再转为输入x.dtype，调用F.linear；true分支才显式调用torch._scaled_mm。
- 因此，若生效fp8_fast=false，则该scaled Linear分支是反量化后按输入类型计算；
  仅凭“optimized/monkey-patched 300层”不能宣称已执行硬件FP8矩阵乘。

完整源码验收仍需补普通FP8实际前向路径、保存源码与固定服务器版本的对应，以及原始参数核对；
不将此初步scaled分支检查标成两条路径全部完成。

## 当前进度

| 项目 | 状态 |
|---|---|
| 普通FP8两步完整链路 | 原始JSON、finite、视频解码及本地归档一致性已验收 |
| scaled FP8两步完整链路 | 原始JSON、finite、视频解码及本地归档一致性已验收 |
| 两种实际前向路径 | 固定源码静态路径和实际参数已核对，不作GPU内核精度实测声明 |
| 主环境baseline/FP8/scaled正式性能 | 待执行 |
| 72段生成与质量评估 | 待执行 |

下一步主环境GPU0运行当期torch baseline total，预热1+正式3，sample-smi；
GPU1不并行生成。随后再测普通FP8及scaled FP8，不复用副本环境A1。
无需重跑两步冒烟，不修改已验收bench.py的week2原始输出路径。

## 2026-10-28：原始产物本机验收

用户将服务器压缩包解压至
`notes/week4/W4-fp8-smoke-artifacts-20261027/`，保留内部notes/week2与outputs/week2。
新增 `scripts/audit_fp8_smoke.py`，用已有项目Python在CPU读取与复核，不运行模型生成、不安装依赖。
两次各session、summary、result及latent/视频，共10份文件齐全。

本机archive SHA256：
`fc760f908e46102ce12a537ea95d67d164b06fd820c244e27882307d65170372`。
解压后的全部10份文件与本机压缩包成员逐字节哈希一致，输入在核验前后未变化。
用户尚未提供服务器压缩包SHA256，故不声称独立完成服务器到本机的哈希对照。
本机完整逐文件哈希、来源、参数与内容检查记录为 `notes/week4/fp8_smoke_audit.json`。

### 原始运行记录

- 两次session/result均status=ok；均stages、2步、预热0、正式1、物理GPU0映射cuda:0。
- 配置及bench哈希与本机原件一致，记录commit为 `f8a1b03794a49239a3539015075f5123d6c07d66`。
- config与effective_args逐项核对，仅将冻结20步临时覆盖为2步、save_path指向独立会话。
- 生效attn_mode=torch、fp8_fast=false、fp8_t5=false、compile=false、blocks_to_swap=0；普通与scaled开关符合各自配置。
- 两次生效参数的差异集合仅为fp8、fp8_scaled与save_path。记录的Python/torch/CUDA/GPU/commit、allocator、tokenizer和environment相同。
- 每次8项阶段函数各调用1次、4次前向；重新聚合所有阶段时间/峰值及summary，与原件完全一致，计时gap均小于2%。
- 主环境解释器使用依据用户执行命令；session未独立记录sys.executable，不声称额外验明该路径或计时全程另一卡占用。

| 两步stages诊断 | 普通FP8 | scaled FP8 |
|---|---:|---:|
| 原始耗时（秒） | 33.126806498039514 | 33.54078897414729 |
| DiT加载（秒） | 0.5687359496951103 | 0.7398340739309788 |
| DiT加载allocated峰值（MiB） | 4138.41552734375 | 1552.0908203125 |
| 去噪（秒） | 9.836989855859429 | 9.913477743044496 |
| 去噪allocated峰值（MiB） | 2883.470703125 | 2916.95654296875 |
| 去噪reserved峰值（MiB） | 4508 | 4508 |
| 整次allocated峰值（MiB） | 14698.353515625 | 14698.353515625 |
| 整次reserved峰值（MiB） | 18666 | 18666 |
| allocated峰值阶段 | VAE解码 | VAE解码 |

单次两步诊断不构成正式时间对照，也不与历史副本环境的20步内存作最终收益判定。
本次sample-smi=false，没有整卡采样峰值；allocated、reserved和阶段峰值不相加。

### latent与视频

两份latent均为float32、形状 `[16, 21, 60, 104]`，逐元素finite，无NaN/Inf。
普通FP8 min/max为-2.3988494873046875/1.9083036184310913；
scaled为-2.4757790565490723/2.5457329750061035。这里只检查有限值，不设事后质量阈值。

两段MP4均完整解码81帧、H264、832×480、16fps；81帧均非全黑、非恒定像素。
抽取第0/40/80帧生成 `notes/week4/fp8_smoke_preview.png`，本机已查看，两者均明显模糊。
这仅说明两步冒烟产物可解码、非黑屏，不代表20步质量通过，也不能把模糊归因于某种FP8配置。

### 固定源码与运算路径核对

本机 `local/musubi-tuner` 可读取固定commit对象。对下列保存源码逐字比较git blob，
仅允许CRLF/LF差异，全部一致；原始及blob SHA256保存在本次审计。

| 保存源码 | 固定commit路径 | 静态依据 |
|---|---|---|
| notes/source/wan_generate_video.py | src/musubi_tuner/wan_generate_video.py | 2195起区分权重类型；644起传递fp8_fast；695起普通权重cast；1531起创建BF16 Accelerator；1455起在autocast中调用模型 |
| notes/source/wan_model.py | src/musubi_tuner/wan/modules/model.py | 标准nn.Linear；1042起加载与scaled前向替换，1073附近普通权重整体cast |
| notes/source/fp8_optimization_utils.py | src/musubi_tuner/modules/fp8_optimization_utils.py | 365起patched前向，378起use_scaled_mm分支，418起反量化并调用F.linear |

普通FP8将权重直接cast为float8_e4m3fn，未启用scaled Linear替换；
主要Linear仍由标准PyTorch路径在BF16 autocast上下文中计算，权重转换交给autocast，
不是通过专门的FP8缩放矩阵乘路径完成。
scaled保存量化权重与scale_weight，生效fp8_fast=false使use_scaled_mm=false，
前向显式反量化，再转为x.dtype并F.linear；本任务外层为BF16 autocast。
不能将两条路径概括为每一个算子都BF16；模型内部仍有显式FP32等计算。

因此本轮主要是低精度权重存储与加载/运行内存的取舍，不预设FP8硬件矩阵乘提速。
上述为固定应用源码+生效参数的静态路径核对，没有新采集FP8内核trace或逐层激活dtype，
不把静态设计表述成完整GPU指令级实测。

## 2026-10-28：F0当期主环境基线收到服务器输出

会话：`W4-F0-torch-total-20261028-003533-43e7c074`。
用户读取session与summary，status=ok，total、20步、warmup1/repeat3、GPU0、sample_smi=true。
生效配置为torch attention、BF16源权重，fp8/fp8_scaled/fp8_fast均false；
记录的配置/bench哈希与此前冻结值一致，commit与两次冒烟相同。
使用主环境解释器依据前述执行命令，session未独立记录解释器路径。

| F0摘要（用户服务器输出） | 数值 |
|---|---:|
| 正式次数 | 3 |
| 均值（秒） | 134.03493864291036 |
| 最小（秒） | 133.4579865988344 |
| 最大（秒） | 135.17331278230995 |
| allocated峰值（各次均相同，MiB） | 14698.353515625 |
| reserved峰值（各次均相同，MiB） | 18666 |

原始session、各轮result、summary及gpu.csv未传回，逐轮聚合、采样整卡峰值及文件哈希尚未本机验收。
不从summary中的allocator值推断整卡峰值。F0作为本组当期基线，不能与副本A1混合统计。
下一步F1普通FP8，同主环境GPU0，20步total、预热1+正式3、sample-smi，标签W4-F1-fp8。
正式性能计划项仍保持未完成，待F1/F2及阶段对照和原件核验后更新。

## 2026-10-28：F1普通FP8收到服务器摘要

会话：`W4-F1-fp8-total-20261028-133106-bf2ccdec`。
用户提供run-03末尾和summary：20步完成、latent/视频保存，run-03打印132.320秒；
summary正式3次均值132.43742388285077秒，范围131.07531786011532–133.9174454039894秒。
三次allocated峰值均14698.353515625 MiB，reserved均18666 MiB，与F0摘要相同。
日志明确将DiT权重cast为float8_e4m3fn；原始session/逐轮result/gpu.csv尚未传回，不替代本机原件核验。

以用户返回的F0/F1摘要计算，均值少1.597514760059596秒，耗时下降1.1918644319416003%，未达到5%门槛。
闭区间重叠于133.4579865988344–133.9174454039894秒，因此当前不确认速度收益。
按冻结规则触发唯一一轮baseline与普通FP8双方追加测量，各预热1+正式3；
保留最初3次与新增3次，双方分别合并6次，不能只取较快轮次或只补测一侧。
范围重叠的初始极值会保留在合并范围中，所以合并范围不可能因追加而变为不相交；
如实执行冻结规则并报告其结果，不将追加描述为保证能确认速度的机会。

当前先完成已安排的F2 scaled FP8首轮，再根据其结果安排追加范围及阶段诊断；
不与F2并行启动确认测量，不开始72段生成。整体质量和省显存结论仍未评估完成，
无整卡CSV就不填整卡峰值，也不以整体峰值相同推断DiT阶段没有收益。

## 2026-10-28：F2 scaled FP8收到服务器摘要

会话：`W4-F2-fp8-scaled-total-20261028-135518-9da4d16a`。
用户提供run-03末尾，20步、latent/视频保存及summary写出完成；
随后粘贴summary主要字段（末尾未完整闭合，不将粘贴文本当成已验收JSON文件）。
measured_runs=3，mean134.91440854842463秒，min/max134.57658117823303–135.53854936733842秒。
三次allocated峰值均14698.353515625 MiB，reserved均18666 MiB。
session/逐轮result/gpu.csv尚未传回，本机聚合及整卡采样仍待核验。

相对F0均值多0.8794699055142701秒，耗时增加0.6561497430586459%；
范围交集为134.57658117823303–135.17331278230995秒。首轮未显示速度收益，不能据这次轻微增加判定稳定变慢。
两种FP8首轮均与F0范围重叠，按冻结规则各触发唯一确认轮次。
同一未变环境可共享基线，因此依次安排：

1. F3：baseline total，标签W4-F3-torch-confirm。
2. F4：普通FP8 total，标签W4-F4-fp8-confirm。
3. F5：scaled FP8 total，标签W4-F5-fp8-scaled-confirm。

每组GPU0、预热1+正式3、sample-smi，GPU1不并行生成；不修改环境、配置或脚本。
最终基线合并F0+F3，普通FP8合并F1+F4，scaled合并F2+F5，各6次，不丢首轮。
确认轮后不再追加total追求通过，接着以20步stages分析阶段内存及时间。
当前只安排F3，F3/F4/F5均尚未收到执行证据。

## 2026-10-28：F3基线确认轮收到服务器输出

会话：`W4-F3-torch-confirm-total-20261028-142252-ac3ea804`。
用户提供session/summary：status=ok，20步total、预热1+正式3、GPU0、sample-smi；
torch attention、FP8各开关关闭，配置/bench哈希、commit及记录runtime与F0一致。
正式mean134.02658165665343秒，min/max133.9448650549166–134.0878832731396秒。
allocated各次14698.353515625 MiB、reserved各次18666 MiB。

F0/F3均3次，按两份摘要等权合并的六次基线mean约134.0307601497819秒，
合并min/max133.4579865988344–135.17331278230995秒；最终精度及逐轮值待原件复算。
原始session/result/summary/gpu.csv尚未传回，不将服务器粘贴输出标为本机原件验收。
下一步F4普通FP8唯一确认轮，随后F5 scaled唯一确认轮；两者未收到执行证据。

## 2026-10-28：F4普通FP8确认轮收到服务器输出

会话：`W4-F4-fp8-confirm-total-20261028-144843-947405db`。
用户返回session/summary：status=ok、total、20步、warmup1/repeat3、GPU0、sample-smi。
fp8=true、fp8_scaled=false、fp8_fast=false、attn_mode=torch，冻结配置/bench哈希及commit与此前一致，记录runtime相符。
正式mean132.23342725541443秒，min/max132.08548581693321–132.50717843603343秒；
allocated各次14698.353515625 MiB、reserved各次18666 MiB。

按两轮各3次摘要合并，F1+F4六次mean132.3354255691326秒，
范围131.07531786011532–133.9174454039894秒；
对应F0+F3基线mean134.03076014978188秒、范围133.4579865988344–135.17331278230995秒。
合并耗时下降约1.2648847016570715%，未达到5%，合并范围仍重叠。
据服务器摘要，本组普通FP8速度门槛未通过；不能仅用F4较窄范围替代全部六次，且不再追加total。
完整原始逐轮数据、summary和gpu.csv尚未传回，最终数值仍需原件复算；
阶段显存及质量尚未评估完成，不据速度未达标单独作最终可用分类。

下一步F5 scaled FP8唯一确认轮，主环境GPU0、20步total、预热1+正式3、sample-smi，
标签W4-F5-fp8-scaled-confirm。F5尚无执行证据；其后统一验收原件并进行阶段测量。

## 2026-10-28：F5摘要完成，total两轮汇总待原件验收

会话：`W4-F5-fp8-scaled-confirm-total-20261028-153815-14aeaba4`。
用户返回session/summary，status=ok、20步total、预热1+正式3、GPU0、sample-smi；
生效fp8=false、fp8_scaled=true、fp8_fast=false、attn_mode=torch，
配置/bench哈希、commit及记录runtime与此前scaled路径一致。
正式mean134.3752808373732秒，min/max133.48830372607335–135.04951727204025秒。
allocated各次14698.353515625 MiB，reserved各次18666 MiB。

下表由用户提供的每轮3次摘要合并，预热全部排除。尚未读取逐轮原件，本表不是最终本机验收。

| 配置 | 纳入会话 | 正式次数 | 均值秒 | min/max秒 | 相对基线耗时 |
|---|---|---:|---:|---|---|
| torch基线 | F0+F3 | 6 | 134.03076014978188 | 133.4579865988344–135.17331278230995 | 参考 |
| 普通FP8 | F1+F4 | 6 | 132.3354255691326 | 131.07531786011532–133.9174454039894 | 少1.2648847016570715% |
| scaled FP8 | F2+F5 | 6 | 134.64484469289891 | 133.48830372607335–135.53854936733842 | 多0.4581668733586078% |

两候选均未达到预登记5%速度门槛，范围均与基线重叠；不再追加total。
scaled均值略高不是稳定变慢的证明，普通FP8均值略低也不等于确认提速。
三配置摘要中的整次allocated/reserved峰值均相同，不能由此否定DiT阶段资源收益；
gpu.csv未传回，不填整卡峰值。质量与最终分类仍待后续。

下一步按六个明确会话目录打包传回notes记录（含预热及正式result/gpu.csv、session/summary），
不需要total视频或latent。原件验收后在主环境依次进行三种配置的20步stages，预热1+正式1，
关注DiT加载/去噪与VAE解码的显存峰值；当前stages尚未执行，不启动72段生成。

## 2026-10-28：F0–F5原始total记录本机验收通过

原件目录：`notes/week4/W4-fp8-total-records-20261028/`。
新增只读入口 `scripts/audit_fp8_total.py`，审计输出 `notes/week4/fp8_total_audit.json`。
六个会话共60份文件，包含6份session、6份summary、24份result及24份gpu.csv。
本机tar SHA256为 `89b7b7fdc2c012d87a5379f234a97ed697560e6504b979f211656c27d3c50f97`，
全部成员与解压文件哈希一致；尚无服务器tar哈希，不声称独立完成跨机器哈希比对。

验收项目：

- 六个session和24份result均status=ok，每个会话1次预热、3次正式，标志和目录正确。
- 固定20步/P01/s42及其余生效参数正确，GPU0、torch attention、fp8_fast=false。
- 各session配置与本机YAML一致，config/bench哈希与本机原件相符，commit固定。
- 六个会话记录的Python/torch/CUDA/GPU、environment、allocator、tokenizer与generation_module字段一致。
- 24轮完整生效参数归一化后仅fp8、fp8_scaled和会话输出路径有差异，未出现其他配置混入。
- total没有stages/profiler/forward插桩，六份summary调用未修改bench.make_summary从原始结果复算，完全一致。
- 24份CSV共31,692行，均可解析、GPU索引0、时间戳非递减、显存与利用率数值合法；
  每轮行数、峰值与result中的external_gpu_sampling一致，interval_ms=100、warning=null。
- 输入文件核验前后哈希不变，预热原件保留但不计正式六次统计。

| 配置 | 正式次数 | 原始六次均值秒 | min/max秒 | 耗时变化 |
|---|---:|---:|---|---|
| torch基线 F0+F3 | 6 | 134.0307601497819 | 133.4579865988344–135.17331278230995 | 参考 |
| 普通FP8 F1+F4 | 6 | 132.33542556913258 | 131.07531786011532–133.9174454039894 | 少1.2648847016571135% |
| scaled FP8 F2+F5 | 6 | 134.64484469289891 | 133.48830372607335–135.53854936733842 | 多0.4581668733585865% |

由原始六次数据复算得到的末尾浮点位与前述两个三次摘要均值等权计算可能略异，
本表及审计的原始数据复算值为最终total统计，不是新增测量或选取较优轮次。
两候选均未达5%且闭区间重叠，唯一确认轮结束，不再追加total。

| 整次峰值口径（MiB） | torch | 普通FP8 | scaled FP8 |
|---|---:|---:|---:|
| allocated，每次正式均相同 | 14698.353515625 | 14698.353515625 | 14698.353515625 |
| reserved，每次正式均相同 | 18666 | 18666 | 18666 |
| 整卡100ms采样，每次正式均相同 | 19627 | 19627 | 19627 |

三种口径不相加；整卡采样可能漏瞬时峰值，且覆盖少量计时外setup/reporting。
本轮整次峰值无下降，不据此推断DiT阶段无收益。total产物视频与latent未传回，
本机不将计时记录验收表述为20步质量验收。解释器路径仍依据用户运行命令，
未收到另一卡/进程全程监控，不声称独立证实全程无并发任务。

下一步按顺序安排F6 torch-stages、F7 fp8-stages、F8 fp8-scaled-stages，
均主环境GPU0、20步、预热1+正式1，GPU1不并行生成。
阶段测量只用于定位时间和内存，不替代上述total速度门槛。
当前仅给F6命令，三项stages均尚未收到执行证据。

## 2026-10-29：F6基线stages收到服务器日志

会话：`W4-F6-torch-stages-stages-20261029-151302-78d70e8b`。
用户返回完整终端日志，stages、20步、预热1+正式1、cuda:0，DiT权重/计算与VAE均打印BF16。
两轮采样、latent保存、VAE解码及MP4保存均结束，打印各自result和最终summary。
预热129.010秒、正式135.730秒，allocated峰值均14698.4 MiB，均为终端舍入值。
日志支持F6完成，不因此前查找命令粘贴错误要求重跑。
原始session/result/summary未传回，阶段聚合、40次前向和生效参数待本机核验，
不将此stages耗时替换已验收total速度统计。
下一步F7普通FP8 stages，同主环境GPU0、20步、预热1+正式1，标签W4-F7-fp8-stages；
其后F8 scaled。F7/F8尚未收到执行证据。

## 2026-10-29：F7普通FP8 stages收到服务器日志

会话：`W4-F7-fp8-stages-stages-20261029-152149-75a61563`。
用户返回完整终端日志，stages、20步、预热1+正式1、cuda:0；
DiT计算精度BF16、权重float8_e4m3fn、VAE BF16，fp8_scaled=false，
两轮均明确执行权重cast，全部键匹配。
预热129.493秒、正式133.552秒，allocated峰值均14698.4 MiB，均为终端舍入值。
两轮20步采样、latent保存、VAE解码和MP4保存结束，打印result及最终summary。
原始session/result/summary未传回，实际参数、40次前向、阶段聚合和峰值待本机核验。
不将单次stages总耗时用于替代正式total速度结论，也不由整次峰值推断DiT阶段显存。
下一步F8 scaled FP8 stages，同主环境GPU0、20步、预热1+正式1，
标签W4-F8-fp8-scaled-stages；F8尚未收到执行证据。

## 2026-10-29：F8 scaled阶段测量收到完整日志

会话：`W4-F8-fp8-scaled-stages-stages-20261029-152931-00dfa303`。
用户提供完整终端日志，stages、20步、预热1+正式1、cuda:0；
两轮均打印fp8_scaled=true、300层Linear优化及300层前向替换，全部键匹配。
预热132.960秒、正式136.515秒，allocated峰值均14698.4 MiB，均为终端舍入值。
两轮采样、latent保存、VAE解码和MP4保存完成，打印result及最终summary。

F6/F7/F8服务器日志均已显示完成，无需重跑；原始阶段JSON尚未传回，
生效参数、各40次前向、阶段聚合和summary仍待本机验收。
不使用单次stages总耗时替代已验收的六次total统计。
下一步仅打包三个notes会话目录传回，含session/summary、warmup-01/result与run-01/result，
暂不需要输出视频或latent。收到后核对DiT加载、去噪及VAE解码的阶段显存，
再完成性能表；72段评测生成尚未启动。

## 2026-10-29：F6–F8阶段原始记录本机验收完成

用户最初指向旧total目录；检查发现新阶段tar已传到本机，但未找到F6–F8解压文件。
新增 `scripts/audit_fp8_stages.py`，先验证tar成员完整集合、普通文件类型及路径边界，
再解压到独立 `notes/week4/W4-fp8-stages-records-20261029/`；
不同内容的已有文件拒绝覆盖，旧total目录与记录不改。
本机tar SHA256为 `8afe8ff030a6773de366e16ad13d5d20625635fd9cbb420f393b9fadb3487882`。
12份原始JSON与tar成员一致；服务器tar哈希未提供，不声称跨机器独立哈希比对。
审计为 `notes/week4/fp8_stages_audit.json`，所有输入哈希保存且前后不变。

三次session及预热/正式共6份result均ok；配置/bench哈希和commit匹配，
生效参数除fp8、fp8_scaled及输出路径外相同，记录runtime与total一致。
每轮20步、40次前向，8个阶段函数各调用一次；全部阶段时间/峰值、计时gap和三份summary重新聚合一致。
每侧只有1次正式stages，预热不入表，以下只用于定位机理，不替代六次total。

### 正式阶段耗时（秒）

| 阶段 | BF16基线 F6 | 普通FP8 F7 | scaled FP8 F8 |
|---|---:|---:|---:|
| T5加载 | 10.644571 | 10.374951 | 10.617930 |
| 文本/输入准备，不含T5 loader | 0.426692 | 0.427154 | 0.423923 |
| DiT加载 | 0.491933 | 0.577264 | 0.709733 |
| 去噪 | 112.293808 | 110.428836 | 113.085364 |
| 保存latent | 0.003620 | 0.003551 | 0.003406 |
| VAE加载 | 0.066029 | 0.059220 | 0.066412 |
| VAE解码，不含loader | 10.321187 | 10.398927 | 10.319693 |
| 保存视频 | 1.118171 | 0.980075 | 0.955761 |
| 其他 | 0.364070 | 0.301789 | 0.332616 |
| stages整次，仅诊断 | 135.730437 | 133.552130 | 136.515232 |

普通FP8本次去噪少1.864972秒，scaled多0.791556秒；每侧单次不建立稳定因果效应。
每次前向均值为2.798622/2.752979/2.818600秒，各40次来自同一次生成，
不能当成40个独立生成样本或等同完整采样步。

### 阶段显存与整次峰值（MiB）

| 口径 | BF16 | 普通FP8 | scaled FP8 |
|---|---:|---:|---:|
| DiT加载allocated峰值 | 2732.646973 | 4139.415527 | 1553.215820 |
| DiT加载reserved峰值 | 2978 | 4434 | 1668 |
| 去噪allocated峰值 | 4183.472168 | 2882.593750 | 2917.081543 |
| 去噪reserved峰值 | 5974 | 4510 | 4508 |
| T5加载allocated峰值 | 10845.601563 | 10845.601563 | 10845.601563 |
| 文本/输入准备allocated峰值 | 11093.609863 | 11093.609863 | 11093.609863 |
| VAE解码allocated峰值 | 14698.353516 | 14698.353516 | 14698.353516 |
| VAE解码reserved峰值 | 18666 | 18666 | 18666 |
| 整次allocated峰值，total与stages均观察到 | 14698.353516 | 14698.353516 | 14698.353516 |
| 整次reserved峰值，total与stages均观察到 | 18666 | 18666 | 18666 |
| 整卡采样峰值，仅此前total | 19627 | 19627 | 19627 |

普通FP8去噪allocated少1300.878418 MiB（31.0957%），scaled少1266.390625 MiB（30.2713%）。
这是当前固定任务阶段资源收益，不能据此宣称整次峰值下降、某张小显存卡可运行或质量保持。
整次allocated峰值三者均位于VAE解码，T5/输入阶段也未下降；整次峰值不随去噪阶段减小。
本次stages没有整卡采样，表内19627来自已验收total，不混作本次阶段实测。
reserved缓存高位延续到保存视频，不表示保存视频阶段新分配同样大小的内存。

普通FP8的DiT加载allocated反而比BF16多1406.768555 MiB（约51.48%）；
scaled比BF16少1179.431152 MiB（约43.16%）。与“两个FP8加载都省内存”的笼统预期不符，应保留负面结果。
普通FP8源码先加载BF16状态再做类型转换，转换时存储重叠是可能解释；
尚未采集分配事件，不能把此假设当成已定位原因，也不把阶段峰值当作最终权重常驻大小。

### 当前性能结论与后续

已确认：两配置均降低本任务去噪阶段allocated峰值；scaled另降低DiT加载峰值，
普通FP8加载峰值升高；整次三种显存峰值无下降。
正式六次total显示普通FP8耗时少1.2649%、scaled多0.4582%，两者速度门槛未达。
不新增配置、不启用fp8_fast、不追加更多total追求通过。

性能记录阶段可收尾，质量尚未完成；尚不能最终称“有条件可用（仅省阶段显存）”，
因为该分类仍要求无新增明显退化/最终失败的证据。
下一步按冻结计划生成两配置各36段，共72段，生成结束再核对覆盖率、首次失败/重试及MP4哈希，
后续自动指标与人工评价分开进行。当前72段生成尚未启动。

## 2026-10-29：72段生成准备

用户明确同意按Day7继续两个配置各36段。
本机调用未修改gen_eval.load_plan作只读核对，没有模型导入、写出或GPU生成：
两配置各分片0/2与1/2各18条、无交集、并集36；20步、torch attention、fp8_fast=false。
命名空间分别为fp8、fp8_scaled，不添加tag或steps诊断参数，避免变更正式数据集身份。
bench哈希仍ffc9af27120d7fd0f903b1482122a393c39a372aa0be0f4eed67873851ea775c，
gen_eval哈希仍ef976d59a23dd40eaeb18e16c30779b116e89f7ded0c84bf235f6645cd54fdae。

计划先普通FP8 GPU0分片0/2，首条成功后GPU1分片1/2；该配置结束并汇总36/36后，
再依同样顺序做scaled FP8。主环境env/bin/python，独立tmux会话，日志项目内保存。
成功样本按脚本核对收据/视频/latent哈希后续跑跳过，最多两次尝试，保留首次失败，
不删除输出或更换样本。并行生成耗时不作为延迟基准。
当前给出GPU0首个完整分片的启动命令，服务器占用/脚本及配置哈希待用户返回，
尚无本批生成启动或成功证据；不能把本机计划核对表述为服务器已生成。

## 2026-10-29：两个配置72段生成本机验收通过

用户依次返回fp8与fp8_scaled汇总，均 `Coverage: {'ok': 36}`。
随后传回 `notes/week4/W4-fp8-eval-72-20261029.tar.gz`，
服务器SHA256与本机独立计算完全一致：
`4412ce239e719935db16148916d33d21b6914ae671ad930a4231d61aa7722869`。
压缩包180232723字节，154份普通文件，无latent或outputs/.attempts产物。

新增 `scripts/audit_fp8_generation.py`，验证压缩包哈希、成员类型和项目路径边界，
全量比较已有文件内容后才解压；不同内容拒绝覆盖。
原件按成员原路径放入 `notes/eval/fp8/`、`notes/eval/fp8_scaled/`、
两个同名JSONL和 `outputs/eval/fp8/`、`outputs/eval/fp8_scaled/`。
本机审计为 `notes/week4/fp8_generation_audit.json`，输入文件核验前后哈希不变。

| 项目 | 普通FP8 | scaled FP8 |
|---|---:|---:|
| 冻结样本覆盖 | 36/36 | 36/36 |
| 本机最终MP4哈希匹配 | 36/36 | 36/36 |
| GPU0 / GPU1最终样本数 | 18 / 18 | 18 / 18 |
| 首次attempt-01成功收据 | 36 | 36 |
| 失败尝试 / 重试样本 | 0 / 0 | 0 / 0 |
| events running / ok | 36 / 36 | 36 / 36 |
| events文件数 | 2 | 2 |
| 跳过 / 失败events | 0 / 0 | 0 / 0 |
| 残留样本锁 | 无 | 无 |

逐项核对dataset指纹、完整生成identity、manifest与最终收据一致；
提示词/种子集合为冻结12×3，配置及脚本哈希、20步、torch attention、
fp8_fast=false等生效参数正确。除FP8模式、提示词/种子及输出目录外，
72条生效参数一致，记录的runtime与已验收主环境total一致。
物理GPU分配逐项符合冻结列表的0/2、1/2分片。
每条收据均有且只有一条对应running和终态ok事件，原始终态events与收据一致。

72段视频均完整解码为81帧、832×480、16fps，全部帧非全黑。
这些检查是生成与可解码性验收，不等于人工质量通过；
没有替用户观看/评分，不由并行生成耗时更新正式速度基准。
latent按计划未传回，本机未独立核验其哈希、内容或finite；
服务器summarize按现有脚本已检查latent/video哈希，这一边界保留。

Week4 FP8计划的冒烟、固定源码核对、正式total/stages性能及72段生成已完成。
下一步按Week5安排复用未修改metrics.py函数，对两个配置各36条计算辅助指标，
再制作随机匿名配对/固定单视频评分包，最后按冻结规则给出各配置质量及适用结论。
当前指标、人工评分和最终质量分类均未完成，不因无生成失败就称质量保持或可用。
