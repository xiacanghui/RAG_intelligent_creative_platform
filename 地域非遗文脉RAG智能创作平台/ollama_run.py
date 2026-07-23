"""
地域非遗文脉RAG智能创作平台 - Ollama本地qwen3.6一键拉取、4bit量化、GPU加速启动脚本
基础工程代码由 Vibe Coding 智能体生成
"""
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 固定最优参数配置（RTX4060 Laptop 8G显存适配）
# ============================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:latest")
# RTX4060 8G显存 4bit量化最优参数
GPU_LAYERS = 33  # GPU加速层数
THREADS = 8      # CPU线程数
CTX_SIZE = 4096  # 上下文窗口大小

# ============================================
# 工具函数
# ============================================
def check_ollama_installed() -> bool:
    """检查Ollama是否已安装"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def check_ollama_running() -> bool:
    """检查Ollama服务是否运行中"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_ollama_service():
    """启动Ollama服务"""
    print("[Ollama] 正在启动Ollama服务...")
    try:
        # Windows平台启动命令
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # 等待服务启动
        for i in range(30):
            time.sleep(1)
            if check_ollama_running():
                print("[Ollama] 服务启动成功")
                return True
        print("[Ollama] 服务启动超时")
        return False
    except Exception as e:
        print(f"[Ollama] 启动失败: {e}")
        return False

def pull_model(model_name: str) -> bool:
    """拉取模型"""
    print(f"[Ollama] 正在拉取模型 {model_name}...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        if result.returncode == 0:
            print(f"[Ollama] 模型 {model_name} 拉取成功")
            return True
        else:
            print(f"[Ollama] 模型拉取失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[Ollama] 模型拉取超时")
        return False

def check_model_exists(model_name: str) -> bool:
    """检查模型是否已存在"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m.get("name") == model_name for m in models)
    except requests.RequestException:
        pass
    return False

def test_model_inference(model_name: str) -> bool:
    """测试模型推理"""
    print(f"[Ollama] 正在测试模型 {model_name} 推理...")
    try:
        payload = {
            "model": model_name,
            "prompt": "你好，请简单介绍一下自己",
            "stream": False,
            "options": {
                "num_gpu": GPU_LAYERS,
                "num_thread": THREADS,
                "num_ctx": CTX_SIZE
            }
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            result = response.json()
            print(f"[Ollama] 推理测试成功，响应长度: {len(result.get('response', ''))}")
            return True
        else:
            print(f"[Ollama] 推理测试失败: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"[Ollama] 推理测试异常: {e}")
        return False

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"[GPU] 检测到GPU: {gpu_info}")
            return gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[GPU] 未检测到NVIDIA GPU，将使用CPU推理")
    return None

# ============================================
# 主流程
# ============================================
def setup_ollama():
    """完整Ollama环境配置流程"""
    print("=" * 50)
    print("地域非遗文脉RAG智能创作平台 - Ollama环境配置")
    print("=" * 50)

    # 1. 检测GPU
    get_gpu_info()

    # 2. 检查Ollama安装
    if not check_ollama_installed():
        print("[错误] Ollama未安装，请先访问 https://ollama.ai 下载安装")
        print("[提示] 安装完成后重新运行本脚本")
        return False

    # 3. 启动Ollama服务
    if not check_ollama_running():
        if not start_ollama_service():
            return False
    else:
        print("[Ollama] 服务已在运行")

    # 4. 检查模型是否存在
    if not check_model_exists(MODEL_NAME):
        print(f"[Ollama] 模型 {MODEL_NAME} 不存在，开始拉取...")
        if not pull_model(MODEL_NAME):
            return False
    else:
        print(f"[Ollama] 模型 {MODEL_NAME} 已存在")

    # 5. 测试推理
    if not test_model_inference(MODEL_NAME):
        print("[警告] 模型推理测试失败，可能会影响正常使用")

    print("=" * 50)
    print("[完成] Ollama环境配置完毕")
    print(f"[模型] {MODEL_NAME}")
    print(f"[GPU加速层数] {GPU_LAYERS}")
    print(f"[上下文窗口] {CTX_SIZE}")
    print(f"[服务地址] {OLLAMA_BASE_URL}")
    print("=" * 50)
    return True

def generate_startup_script():
    """生成一键启动批处理脚本（Windows）"""
    script_content = f"""@echo off
chcp 65001 >nul
echo ============================================
echo 地域非遗文脉RAG智能创作平台 - 启动脚本
echo ============================================

:: 启动Ollama服务
echo [1/2] 启动Ollama服务...
start /B ollama serve
timeout /t 5 /nobreak >nul

:: 启动FastAPI服务
echo [2/2] 启动FastAPI服务...
python main.py

pause
"""
    script_path = Path(__file__).parent / "start.bat"
    script_path.write_text(script_content, encoding='utf-8')
    print(f"[脚本] 已生成启动脚本: {script_path}")

    # 生成Linux/Mac版本
    shell_script = f"""#!/bin/bash
echo "============================================"
echo "地域非遗文脉RAG智能创作平台 - 启动脚本"
echo "============================================"

# 启动Ollama服务
echo "[1/2] 启动Ollama服务..."
ollama serve &
sleep 5

# 启动FastAPI服务
echo "[2/2] 启动FastAPI服务..."
python main.py
"""
    shell_path = Path(__file__).parent / "start.sh"
    shell_path.write_text(shell_script, encoding='utf-8')
    print(f"[脚本] 已生成启动脚本: {shell_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ollama环境配置脚本")
    parser.add_argument("--setup", action="store_true", help="执行完整环境配置")
    parser.add_argument("--script", action="store_true", help="生成启动脚本")
    parser.add_argument("--test", action="store_true", help="测试模型推理")
    args = parser.parse_args()

    if args.setup or (not args.script and not args.test):
        setup_ollama()
    elif args.script:
        generate_startup_script()
    elif args.test:
        test_model_inference(MODEL_NAME)
