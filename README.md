## JIRA 可视化 Web 应用

本项目是一个基于 **Streamlit** 的本地 Web 可视化应用，用于从 JIRA 拉取 Issue 数据，并按时间、reporter、assignee、label、status 等维度进行过滤和可视化展示（修复率、总数、修复时间等）。

### 1. 环境准备

- 安装 Python 3.10+（建议使用 virtualenv 或 conda 创建虚拟环境）
- 在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 启动方式

在项目根目录执行：

```bash
streamlit run app.py
```

浏览器会自动打开（或访问 `http://localhost:8501`）。

### 3. JIRA 认证与配置

左侧侧边栏需要填写：

- **JIRA Base URL**：例如 `https://your-domain.atlassian.net/`
- **JIRA 账号邮箱**（仅作展示，可选）
- **JIRA Personal Access Token (PAT)**：建议本地安全存储，应用不会持久化
- **Project Key**：如 `ABC`，多个项目用逗号分隔

你也可以选择性通过环境变量预填部分信息（可选）：

- `JIRA_BASE_URL`（当前示例默认值：`https://jira.tools.3stripes.net/`）
- `JIRA_EMAIL`（当前示例默认值：`peter.chen2@adidas.com`）
- `JIRA_PAT`

你可以参考 `.env.example`，在本地复制为 `.env` 并填入你的真实 PAT，例如：

```bash
cp .env.example .env
# 然后编辑 .env，把 JIRA_PAT 改成你的真实值
```

> 出于安全考虑，本仓库不会包含任何真实 PAT/Token；`.env` 已加入 `.gitignore`，防止误提交。

### 4. 过滤条件（结构化）

侧边栏可以按以下条件过滤：

- 时间：创建日期起止（`created`）
- `reporter`：显示名，多个用逗号分隔
- `assignee`：显示名，多个用逗号分隔
- `label`：多个用逗号分隔
- `status`：多个状态名用逗号分隔
- 最大返回条数 `max_results`（默认 1000）

应用会构建对应的 JQL，并调用 JIRA REST API `/rest/api/2/search` 接口拉取数据。

### 5. 自然语言筛选（可选）

侧边栏有一块「自然语言筛选」区域，你可以直接用中文/英文描述需求，例如：

- `帮我搜索整个2025年GINFOSEC项目下，assign给Peter.chen2@adidas.com的ticket`
- `Find all tickets in project GINFOSEC in 2024 assigned to peter.chen2@adidas.com`

应用会调用阿里云 Qwen（DashScope）接口，把这段描述转换成对应的 JQL（例如：
`project = GINFOSEC AND assignee = "peter.chen2@adidas.com" AND created >= "2025-01-01" AND created <= "2025-12-31"`），然后用这个 JQL 去查 JIRA。

要启用自然语言筛选，需要在环境变量或 `.env` 中配置：

- `DASHSCOPE_API_KEY`：阿里云 DashScope/Qwen 的 API Key
- （可选）`DASHSCOPE_BASE_URL`：兼容模式 chat/completions 端点，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- （可选）`QWEN_MODEL`：模型名，默认 `qwen-plus`

如果不配置这些环境变量，仍然可以使用结构化过滤。

### 6. 可视化内容

- 原始 Issue 表格（带 key、summary、status、assignee、reporter、创建/解决时间、修复用时等）
- 总览指标：总数、已解决数、修复率、平均修复用时（天）
- 按日期聚合：
  - 每日创建量柱状图
  - 每日修复率折线图
- 按 `status` 聚合：状态分布柱状图
- 按 `assignee` 聚合：工作量分布柱状图
- 修复用时（天）分布直方图

### 7. 安全提示

- 不要把真实的 JIRA `PAT / API Token` 提交到任何代码仓库或分享给他人。
- 如需在团队内共享本项目，建议：
  - 将 Token 通过环境变量或本地 `.env` 文件（不纳入版本控制）进行管理；
  - 或者让每位使用者在 UI 中自行输入 Token。

