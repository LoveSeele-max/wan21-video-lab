# attention 后端兼容性检查

2026-10-22，依据用户提供的服务器终端输出。

- Python 3.11.17；PyTorch 2.9.1+cu128；CUDA 构建 12.8。
- 两张 NVIDIA RTX 5880 Ada Generation，计算能力均为 8.9。
- Triton 3.5.1；sageattention 未安装。
- nvcc `/usr/local/cuda/bin/nvcc`，CUDA Toolkit 12.8，V12.8.93。
- GCC `/usr/bin/gcc` 13.3.0；g++ `/usr/bin/g++` 存在；ninja 未找到。
- 官方 SageAttention setup.py 明列 sm_89，要求该架构 CUDA >=12.4。本次版本满足这一构建条件，但完整安装、ABI、运行兼容性仍未验证。
- 核对来源：https://github.com/thu-ml/SageAttention ，https://raw.githubusercontent.com/thu-ml/SageAttention/main/setup.py 。这是要求核对，不将 main 作为已冻结安装版本。

## 进度

- [x] 服务器只读环境检查完成，依据用户输出记录。
- [x] 在项目目录完整复制 Conda 主环境为 `env-sagecheck`，核对 Python/torch/triton。
- [x] 副本安装 ninja，确认解释器路径及 pip 依赖检查通过。
- [x] 获取官方 v2.2.0 源码并记录完整 commit，依据用户克隆输出。
- [x] 副本内编译并安装固定源码 SageAttention 2.2.0，依据用户安装输出。
- [x] 构建日志和安装前后依赖快照已传回本机并核对；源码 commit 依据服务器克隆输出记录，未传回完整源码。
- [x] 安装后 pip check、可导入 API 和 GPU 0 BF16 小张量 CUDA 运算通过。
- [x] 核对安装前后 pip freeze 全部记录，仅新增 SageAttention；不把安装成功当加速证据。

主环境 `/home/user/projects/wan21-video/env` 保留。后端安装、API 和小张量检查通过后，选定 SageAttention 2.2.0 作为候选；完整模型和性能质量对照仍待后续实验。

## 独立环境复制结果

2026-10-22，用户返回 Conda 完整输出：使用 `--clone` 和 `--copy`，目标为 `/home/user/projects/wan21-video/env-sagecheck`，事务执行完成。副本 Python 的检查输出为 torch `2.9.1+cu128`、Triton `3.5.1`、CUDA available `True`。这是依据用户服务器输出的记录；SageAttention 尚未安装、未验证运行。

## 构建工具与待获取源码

2026-10-22，用户返回副本环境安装结果：ninja 包版本 `1.13.2`，可执行程序版本 `1.13.2.git.kitware.jobserver-pipe-1`；`python -m pip check` 为 `No broken requirements found.`。解释器为 `/home/user/projects/wan21-video/env-sagecheck/bin/python`，torch 仍为 `2.9.1+cu128`。

下一步选用官方 SageAttention `v2.2.0` 标签，在项目 `third_party/SageAttention-v2.2.0` 获取源码并记录完整 commit。当前仅选定待检查标签，服务器获取结果、commit、构建依赖及编译运行尚未验收。

## 源码获取结果与构建尝试

2026-10-22，用户返回服务器输出：直连 GitHub SSL connection timeout；通过单条命令的 `http.proxy=http://127.0.0.1:7897` 克隆成功，v2.2.0 检出 commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`。未修改 Git 全局代理设置。

下一步在 env-sagecheck 使用现有构建工具执行 `pip install --no-build-isolation --no-deps -v`，仅面向 sm_89，限制构建并发，并保存依赖快照和日志。关闭构建隔离意味着本次尝试复用副本工具版本，不代表已满足 pyproject.toml 中全部构建工具版本范围；如失败，先依据日志定位，不自动更新 torch/Triton。编译、安装及 CUDA 运算尚未验收。

## 编译安装结果

用户返回日志显示 `Successfully built sageattention` 与 `Successfully installed sageattention-2.2.0`；wheel 内包含 `_fused`、`_qattn_sm80` 和 `_qattn_sm89` 扩展。构建 wheel 为 `sageattention-2.2.0-cp311-cp311-linux_x86_64.whl`，大小 15526355 字节，SHA256 `91a9bb4a4276e812a3cd3d4ee7e1879dcf997217b768cb5ea1900d4a32bcca49`。这是日志提供的产物信息，不代表临时 wheel 文件已永久留存。

安装项已勾选；依赖复核、从源码目录外导入、BF16 小张量 CUDA 运算与日志快照验收仍待进行。尚无模型推理、加速或质量结论。

## 安装后导入和小张量检查通过

用户返回安装后 `No broken requirements found.`；Python 路径属于 env-sagecheck，torch 保持 `2.9.1+cu128`、Triton 保持 `3.5.1`、sageattention 为 `2.2.0`。导入路径为副本 `site-packages/sageattention/__init__.py`，没有从源码工作目录误导入。

GPU 0 RTX 5880 Ada 上，导入 `sageattn` 和 `sageattn_varlen` 成功；`sageattn` 对 BF16、HND、形状 `(1, 12, 256, 128)` 的非因果张量运行，CUDA 同步以及输出形状、dtype、device、有限值断言均通过。`sageattn_varlen` 仅导入，未执行；GPU 1 未做此运算检查。该检查不验证数值误差、长序列、完整 Wan 模型、性能收益或质量。

选定 attention 候选为 v2.2.0 / commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`。下一步传回服务器 `notes/sagecheck/`，核对安装前后完整依赖差异、构建日志和快照留存；这些归档验收尚未勾选。

## 本机归档与依赖差异验收

2026-10-22，实际读取传回的 `notes/sagecheck/build-20261022-150518/before.txt`、同目录 `install.log` 和 `notes/sagecheck/after.txt`。安装前 71 条非空记录，安装后 72 条；没有删除或修改原有条目，唯一新增为 `sageattention @ file:///home/user/projects/wan21-video/third_party/SageAttention-v2.2.0`。torch、Triton、CUDA 相关 Python 包及 musubi-tuner 可编辑安装 commit 均保持一致。该比较从安装 ninja 后开始，不包含 ninja 安装前的变化；pip freeze 也不是 Conda、编译工具或系统库的完整快照。

构建日志确认副本解释器、`/usr/local/cuda/bin/nvcc`、`compute_89/sm_89` 编译及最终安装成功，wheel 信息与用户贴出的记录相同。保留了打包文件匹配、源码行尾及 PyTorch 未定义 CUDA 12.8 g++ 版本边界等警告；它们没有阻止本次构建和小张量检查，不据此承诺其他路径兼容。

三份归档文件已计算 SHA256，详见 `notes/sagecheck/archive_audit.json`。源码仍保留在服务器，完整源码和临时 wheel 未传回；源码 commit 来自用户克隆输出。构建及依赖归档验收完成，完整模型、GPU 1、性能及质量仍未验证。
