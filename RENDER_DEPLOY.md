# Render部署指南 - 地域非遗文脉RAG智能创作平台

## 第一步：创建GitHub仓库

1. 打开 https://github.com/new
2. 仓库名填写：`RAG_intelligent_creative_platform`
3. 选择 **Public**（免费）
4. 点击 **Create repository**

## 第二步：推送代码到GitHub

在项目根目录运行以下命令（替换 `你的GitHub用户名`）：

```bash
cd E:\programming\opencode\project\RAG_intelligent_creative_platform

# 关联远程仓库
git remote add origin https://github.com/你的GitHub用户名/RAG_intelligent_creative_platform.git

# 推送代码
git push -u origin main
```

## 第三步：配置Render

1. 打开 https://render.com 注册/登录
2. 点击 **New** → **Web Service**
3. 连接你的GitHub仓库（授权后选择 `RAG_intelligent_creative_platform`）

### Render配置项：

| 配置项 | 值 |
|--------|-----|
| **Name** | `rag-creative-platform` |
| **Region** | Singapore (或离你最近的) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `cd "地域非遗文脉RAG智能创作平台" && python main.py` |

### 环境变量（Environment Variables）：

点击 **Add Environment Variable**，逐个添加：

| Key | Value |
|-----|-------|
| `ZHIPU_API_KEY` | `3f642da3fa3e4596b0cee8cbcd86c66c.S5cHQMZf7GXNumpW` |
| `SILICONFLOW_API_KEY` | `sk-csqzuhvqfxtnuzbtipxbzedmnphrrrdmuxiwojmfwuesormj` |
| `AGNES_API_KEY` | `sk-rtYlyfsdTbfCi4HIaQtluGvX7DpU9oyAXXYAMcBf2djJyUgW` |
| `OLLAMA_BASE_URL` | （留空） |
| `OLLAMA_MODEL` | （留空） |
| `FASTAPI_HOST` | `0.0.0.0` |
| `FASTAPI_PORT` | `8000` |

4. 点击 **Create Web Service**

## 第四步：访问你的网站

Render部署成功后会给你一个网址，格式类似：
```
https://rag-creative-platform.onrender.com
```

打开这个网址就能使用你的平台了！

---

## 注意事项

1. **首次部署需要5-10分钟**，后续更新代码会自动重新部署
2. **免费tier有冷启动限制**：15分钟无访问后会休眠，下次访问需要30-60秒唤醒
3. **本地Ollama模型不可用**：Render服务器没有GPU，只能使用云端模型（智谱/硅基流动）
4. **API密钥安全**：密钥配置在Render环境变量中，不会暴露在代码里
