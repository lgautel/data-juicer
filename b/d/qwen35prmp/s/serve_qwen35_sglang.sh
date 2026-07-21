#!/usr/bin/env bash
# =============================================================================
# serve_qwen35_sglang.sh — 用 SGLang 启动 Qwen3.5-4B OpenAI 兼容推理服务
# =============================================================================
#
# 参考: Qwen3.5-4B/README.md 「Serving Qwen3.5 -> SGLang」章节。
# 硬件背景: NVIDIA RTX PRO 6000 Blackwell (sm_120)；系统默认 CUDA 常为 12.8。
#
# -----------------------------------------------------------------------------
# 一、如何使用
# -----------------------------------------------------------------------------
#
# 1) 准备环境（只需做一次；已完成后可跳过）
#    - 独立 venv（勿污染 rlinf）: /home/luogang/VENV/sglng
#    - 安装:
#        uv pip install --prerelease=allow --python /home/luogang/VENV/sglng/bin/python \
#            "sglang[all]==0.5.15.post1" openai ninja "nvidia-cuda-cccl==13.0.85"
#      说明: --prerelease=allow 因依赖 flash-attn-4 的 beta；cccl 补 nv/target 头文件。
#
# 2) 启动服务（本脚本会 source venv 的 activate，可不先手动 activate）
#      ./serve_qwen35_sglang.sh                 # 标准
#      MODE=tool ./serve_qwen35_sglang.sh       # 工具调用
#      MODE=mtp  ./serve_qwen35_sglang.sh       # MTP 投机解码
#    或先激活再手写启动:
#      source /home/luogang/VENV/sglng/bin/activate
#      python -m sglang.launch_server --model-path ... --reasoning-parser qwen3 ...
#
# 3) 验证
#      /home/luogang/VENV/sglng/bin/python test_client.py
#      /home/luogang/VENV/sglng/bin/python test_client.py --no-thinking
#      curl -s http://localhost:8000/health
#      curl -s http://localhost:8000/v1/models
#
# 4) 停止
#      pkill -f 'sglang[.]launch_server'   # 用正则括号，避免 pkill -f 误杀自身 shell
#
# 常用可覆盖环境变量:
#   VENV_DIR / VENV_PY   虚拟环境目录或 python 路径
#   MODEL_PATH           模型权重目录(默认 /home/luogang/CKPT/VLM/Qwen3.5-4B)
#   HOST / PORT          监听地址/端口(默认 0.0.0.0:8000)
#   TP_SIZE              张量并行 GPU 数(默认 1)
#   CUDA_VISIBLE_DEVICES 指定 GPU(默认 0)
#   MEM_FRACTION_STATIC  静态显存占用比例(默认 0.8)
#   CONTEXT_LENGTH       上下文长度(默认 262144)
#   MODE                 standard | tool | mtp
#
# -----------------------------------------------------------------------------
# 二、文件增删改一览（及相关原因）
# -----------------------------------------------------------------------------
#
# [新建] /home/luogang/VENV/sglng/
#   原因: 用户要求独立环境装 sglang；原 /home/luogang/VENV/rlinf 为 RL 训练环境
#         (torch 2.8 + sglang 0.4.6)，升级会破坏训练栈。sglang 支持 Qwen3.5 最早
#         自 v0.5.9，且强制 torch>=2.9，最终选正式版 0.5.15.post1（含 qwen3_5.py）。
#
# [修改] /home/luogang/VENV/sglng/bin/activate
#   原因: 把 Blackwell + pip CUDA13 所需的 PATH/LD_LIBRARY_PATH/CUDA_HOME/
#         LIBRARY_PATH/NVCC_APPEND_FLAGS 等一次性写进 activate；deactivate 用
#         _SGLNG_ENV_ACTIVE 还原，避免重复在每个脚本里抄环境变量。
#
# [修改] 本文件 serve_qwen35_sglang.sh
#   原因: 默认 VENV/MODEL 路径从旧机器路径改为本机 sglng + Qwen3.5-4B；
#         环境变量逻辑改为 source activate，脚本本身只保留服务参数。
#
# [修改] test_client.py（仅注释中的 python 路径）
#   原因: 示例命令改为 /home/luogang/VENV/sglng/bin/python。
#
# [包安装] 见上文「准备环境」；未改模型权重文件本身。
#
# -----------------------------------------------------------------------------
# 三、遇到的 Bug → Fix（按时间线）
# -----------------------------------------------------------------------------
#
# Bug-1  旧 sglang 无 Qwen3.5
#   现象: rlinf 内 sglang==0.4.6.post5 仅有 qwen3.py，无 qwen3_5 架构。
#   Fix:  新建 sglng venv，装 sglang[all]==0.5.15.post1（正式版已含 qwen3_5.py，
#         不必从 git main 编译）。
#
# Bug-2  uv/pip 解析 flash-attn-4 预发布失败
#   现象: No solution found ... flash-attn-4==4.0.0b15，pre-releases weren't enabled。
#   Fix:  uv pip install --prerelease=allow ...
#
# Bug-3  FlashInfer: "requires GPUs with sm75 or higher"（实为 sm120 探测失败）
#   现象: Failed to get device capability: SM 12.x requires CUDA >= 12.9；
#         TARGET_CUDA_ARCHS 为空后误报 sm75。
#   根因: 系统 CUDA_HOME=/usr/local/cuda-12.8，nvcc 12.8 < 12.9；flashinfer 用
#         which nvcc / CUDA_HOME 判定版本，拒绝 sm120。
#   Fix:  activate 将 CUDA_HOME/CUDA_PATH/PATH 指向 venv 内
#         site-packages/nvidia/cu13（pip 自带 nvcc 13.2）。
#
# Bug-4  FileNotFoundError: 'ninja'
#   现象: flashinfer JIT 调 ninja 失败。
#   根因: 脚本用绝对路径 python 启动，venv/bin 不在 PATH。
#   Fix:  activate 已把 $VIRTUAL_ENV/bin 加入 PATH；并确保装了 ninja。
#
# Bug-5  "CUDA compiler and CUDA toolkit headers are incompatible"
#   现象: cccl cuda_toolkit.h 报 nvcc 与头文件大版本不一致。
#   根因: pip 拆分包：nvcc 13.2，CUDART_VERSION 头文件 13.0。
#   Fix:  export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ..."
#         （官方宏，见 cuda_toolkit.h；次版本差对 sm120 编译无实质影响）。
#
# Bug-6  /usr/bin/ld: cannot find -lcudart
#   现象: flashinfer CUDA 目标已编译完，链接阶段失败。
#   根因: pip CUDA 布局是 lib/ 不是 lib64/，且只有 libcudart.so.13 无
#         libcudart.so；链接命令却写 -L$CUDA_HOME/lib64 -lcudart。
#   Fix:  必要时 ln -sf libcudart.so.13 libcudart.so；LIBRARY_PATH 加入
#         $CUDA_HOME/lib 与 /usr/lib/x86_64-linux-gnu（系统 libcuda）。
#
# Bug-7  fatal error: nv/target: No such file or directory
#   现象: sgl-kernel / tvm-ffi JIT 编译 activation 时，cuda_fp16.h 找不到 nv/target。
#   根因: pip 的 nvidia-cuda-* 拆分包未带 CCCL/libcu++ 头文件。
#   Fix:  uv pip install nvidia-cuda-cccl==13.0.85（把头装进 cu13/include/nv/）。
#
# Bug-8  activate 开头 deactivate nondestructive 误清系统 CUDA_HOME
#   现象: 若在 deactivate 里无条件 unset CUDA_HOME，source activate 会先清掉
#         用户原有 CUDA_HOME，再设置；deactivate 后无法还原。
#   Fix:  用 _SGLNG_ENV_ACTIVE=1 标记；仅当本会话真正配置过才还原/清理。
#
# Bug-9  pkill -f sglang.launch_server 误杀启动 shell
#   现象: 命令行自身匹配模式，被 -9 杀掉，服务其实没起来。
#   Fix:  停止时用 pkill -f 'sglang[.]launch_server'。
#
# -----------------------------------------------------------------------------
# 四、activate 里具体导出的变量（source 后生效）
# -----------------------------------------------------------------------------
#   PATH              += $VIRTUAL_ENV/bin 与 $CUDA_HOME/bin
#   LD_LIBRARY_PATH   += site-packages/nvidia/**/lib
#   CUDA_HOME/PATH     = .../nvidia/cu13
#   LIBRARY_PATH       = $CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu
#   NVCC_APPEND_FLAGS  = -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
#
# =============================================================================

set -euo pipefail

VENV_DIR="${VENV_DIR:-/home/luogang/VENV/sglng}"
VENV_PY="${VENV_PY:-${VENV_DIR}/bin/python}"
ACTIVATE="${VENV_DIR}/bin/activate"
MODEL_PATH="${MODEL_PATH:-/home/luogang/CKPT/VLM/Qwen3.5-4B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
MODE="${MODE:-standard}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "${ACTIVATE}" ]]; then
    echo "[ERROR] 找不到虚拟环境 activate: ${ACTIVATE}" >&2
    exit 1
fi
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[ERROR] 找不到虚拟环境 Python: ${VENV_PY}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] 找不到模型目录: ${MODEL_PATH}" >&2
    exit 1
fi

# 加载 venv 内的 CUDA/PATH/LIBRARY_PATH 等配置（见 bin/activate）
# shellcheck source=/dev/null
source "${ACTIVATE}"

# 公共参数
COMMON_ARGS=(
    --model-path "${MODEL_PATH}"
    --served-model-name "Qwen/Qwen3.5-4B"
    --host "${HOST}"
    --port "${PORT}"
    --tp-size "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --context-length "${CONTEXT_LENGTH}"
    --reasoning-parser "${REASONING_PARSER}"
)

case "${MODE}" in
    standard)
        EXTRA_ARGS=()
        ;;
    tool)
        EXTRA_ARGS=(--tool-call-parser "${TOOL_CALL_PARSER}")
        ;;
    mtp)
        # 多 Token 预测(投机解码): 见 README 推荐配置
        EXTRA_ARGS=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 4
        )
        ;;
    *)
        echo "[ERROR] 未知 MODE=${MODE} (可选: standard|tool|mtp)" >&2
        exit 1
        ;;
esac

echo "=========================================================="
echo " 启动 SGLang 服务 (Qwen3.5-4B)"
echo "   MODE          = ${MODE}"
echo "   MODEL_PATH    = ${MODEL_PATH}"
echo "   服务地址      = http://${HOST}:${PORT}/v1"
echo "   GPU           = CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (tp=${TP_SIZE})"
echo "   CONTEXT_LEN   = ${CONTEXT_LENGTH}"
echo "   MEM_FRACTION  = ${MEM_FRACTION_STATIC}"
echo "   CUDA_HOME     = ${CUDA_HOME:-<unset>}"
echo "=========================================================="

exec "${VENV_PY}" -m sglang.launch_server "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
