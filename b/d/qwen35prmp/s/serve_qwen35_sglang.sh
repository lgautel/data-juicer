#!/usr/bin/env bash
# 使用 SGLang 启动 Qwen3.5-4B 的 OpenAI 兼容推理服务
#
# 依赖: 已在虚拟环境 /mnt/r/VENV/dj 中安装 sglang[all]（main 分支）。
# 参考: Qwen3.5-4B/README.md 的 "Serving Qwen3.5 -> SGLang" 章节。
#
# 用法:
#   ./serve_qwen35_sglang.sh              # 标准模式
#   MODE=tool ./serve_qwen35_sglang.sh    # 支持工具调用
#   MODE=mtp  ./serve_qwen35_sglang.sh    # 多 Token 预测(投机解码)加速
#
# 常用可覆盖环境变量:
#   MODEL_PATH  模型权重目录(默认本地 Qwen3.5-4B)
#   PORT        服务端口(默认 8000)
#   TP_SIZE     张量并行 GPU 数(默认 1)
#   CUDA_VISIBLE_DEVICES  指定使用的 GPU(默认 0)
#   MEM_FRACTION_STATIC   静态显存占用比例(默认 0.8)
#   CONTEXT_LENGTH        上下文长度(默认 262144)
#   HOST        监听地址(默认 0.0.0.0)

set -euo pipefail

VENV_PY="${VENV_PY:-/mnt/r/VENV/dj/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/r/share/zwy/Projects/starVLA/playground/Pretrained_models/Qwen3.5-4B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
MODE="${MODE:-standard}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "[ERROR] 找不到虚拟环境 Python: ${VENV_PY}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] 找不到模型目录: ${MODEL_PATH}" >&2
    exit 1
fi

# torch 2.11(cu13) 与 deep_gemm 等依赖通过 pip 提供 CUDA 运行库(如 libnvrtc.so.13),
# 需将 site-packages/nvidia/**/lib 加入 LD_LIBRARY_PATH, 否则会报
# "libnvrtc.so.13: cannot open shared object file".
NVIDIA_LIB_DIRS="$("${VENV_PY}" - <<'PY'
import os, glob, sysconfig
sp = sysconfig.get_paths()["purelib"]
dirs = sorted(set(os.path.dirname(p) for p in glob.glob(os.path.join(sp, "nvidia", "**", "lib"), recursive=True)
                  if os.path.isdir(p)))
# 上面 glob 已到 lib 目录本身
dirs = sorted(set(p for p in glob.glob(os.path.join(sp, "nvidia", "**", "lib"), recursive=True)
                  if os.path.isdir(p)))
print(":".join(dirs))
PY
)"
if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

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
echo "=========================================================="

exec "${VENV_PY}" -m sglang.launch_server "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
