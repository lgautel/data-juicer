# Multi-agent collaboration in cursor
_Exported on 7/13/2026 at 12:40:14 GMT+8 from Cursor (3.10.20)_

---

**User**

在 cursor 里怎么些 multi agent 程序, 调用 cursor 对话界面中能调用的多个VLM模型一起协作在后台完成任务

---

**Cursor**

你想在 Cursor 里用多 Agent / 多 VLM 后台协作。我先查一下 Cursor SDK 和相关能力的官方说明。

接着查官方文档里多 Agent 编排和模型选择的具体写法。

在 Cursor 里做「多模型协作」，核心不是去调聊天框里的模型下拉菜单，而是用 **Subagents（IDE 内）** 或 **Cursor SDK（程序化/后台）** 编排多个 Agent，每个 Agent 指定不同模型。

---

## 两条路怎么选

| 场景 | 做法 |
|------|------|
| 在对话里让主 Agent 自动分工 | `.cursor/agents/*.md` Subagents |
| 脚本 / CI / 后台长期跑、可并行多模型 | `@cursor/sdk`（或 Python `cursor-sdk`） |
| 定时 / webhook 触发 | Cursor Automations |

聊天界面本身**不能**被你的程序直接“点选模型”；SDK/Cloud Agent 走的是同一套 Cursor 托管模型与计费，和 IDE 对话是同一池子。

---

## 1. IDE 内：用 Subagents 多模型协作

在项目里建：

```text
.cursor/agents/planner.md
.cursor/agents/implementer.md
.cursor/agents/reviewer.md
```

每个文件用 frontmatter 指定模型和职责，例如：

```markdown
---
name: reviewer
description: 安全与代码质量审查，只读分析，完成后给出问题清单
model: claude-4.5-opus-high-thinking   # 或 inherit / 具体模型 ID
readonly: true
---

你是代码审查专家。只分析、不改代码，输出按严重程度排序的问题列表。
```

主 Agent 会根据 `description` 决定何时委派。典型编排：

1. **Planner**（强推理模型）出方案  
2. **Implementer**（快/便宜模型）改代码  
3. **Verifier**（另一模型）对照需求验收  

也可设 `background: true`，子 Agent 异步跑，父 Agent 不等待。

可用模型 ID 以 [Models](https://cursor.com/docs/models-and-pricing) 和你账号权限为准；不确定时用 SDK 的 `Cursor.models.list()` 列出（见下）。

---

## 2. 程序化：Cursor SDK 多 Agent 后台协作（推荐做“多 Agent 程序”）

### 准备

```bash
export CURSOR_API_KEY="cursor_..."   # Dashboard → Integrations / API Keys
npm install @cursor/sdk              # 需要 Node ≥ 22.13
```

### 先列出你账号能用的模型（≈ 对话里能选的）

```typescript
import { Cursor } from "@cursor/sdk";

const models = await Cursor.models.list({ apiKey: process.env.CURSOR_API_KEY! });
console.log(models.map((m) => ({ id: m.id, name: m.displayName })));
```

不要写死生僻 ID；以 `list()` 返回为准。

### 模式 A：一个父 Agent + 多个不同模型的 Subagent（自动委派）

```typescript
import { Agent } from "@cursor/sdk";

await using agent = await Agent.create({
  apiKey: process.env.CURSOR_API_KEY!,
  model: { id: "composer-2.5" },          // 编排者
  local: { cwd: process.cwd() },          // 或用 cloud: { repos: [...] } 做后台
  agents: {
    "vision-analyst": {
      description: "看截图/UI，描述布局与缺陷",
      prompt: "你是视觉分析专家。只输出结构化观察，不改代码。",
      model: "gemini-3.5-flash",           // 示例：换成 list() 里的真实 ID
    },
    "implementer": {
      description: "根据计划实现代码改动",
      prompt: "按给定计划改代码并跑相关测试。",
      model: "composer-2.5",
    },
    "reviewer": {
      description: "审查改动的正确性与风险",
      prompt: "只审查、列问题，不直接改。",
      model: "claude-4.5-opus-high-thinking",
    },
  },
});

const run = await agent.send(
  "分析这张 UI 截图，修导航栏问题，再做一次审查"
);
await run.wait();
```

父 Agent 通过内置 Agent 工具委派；子 Agent 可再嵌套一层（最多约两层：父 → 子 → 孙）。

### 模式 B：你自己当编排器，并行起多个独立 Agent（显式多模型）

适合「多个 VLM 同时看图再汇总」：

```typescript
import { Agent } from "@cursor/sdk";

const apiKey = process.env.CURSOR_API_KEY!;
const image = { data: base64Png, mimeType: "image/png" as const };

async function askModel(modelId: string, prompt: string) {
  await using agent = await Agent.create({
    apiKey,
    model: { id: modelId },
    local: { cwd: process.cwd() },
  });
  const run = await agent.send({ text: prompt, images: [image] });
  return (await run.wait()).result;
}

const [a, b, c] = await Promise.all([
  askModel("gemini-3.5-flash", "描述图中缺陷，JSON 输出"),
  askModel("gpt-5.5-medium", "描述图中缺陷，JSON 输出"),
  askModel("claude-sonnet-5-thinking-xhigh", "描述图中缺陷，JSON 输出"),
]);

// 再用一个合成 Agent 合并结果
const consensus = await Agent.prompt(
  `综合以下三份观察，给出最终结论与修复步骤:\nA:${a}\nB:${b}\nC:${c}`,
  { apiKey, model: { id: "composer-2.5" }, local: { cwd: process.cwd() } }
);
```

要点：

- **VLM**：`agent.send({ text, images })` 传图；推理仍走 Cursor 托管模型，不是本机本地 VLM。  
- **后台**：用 `cloud: { repos: [...] }`，进程退出后云端仍可跑；可用 `Agent.resume("bc-...")` 续跑。  
- **本地**：`local: { cwd }`，agent loop 在本机，模型仍在云端。

---

## 3. 后台长期跑的推荐形态

```typescript
const agent = await Agent.create({
  apiKey: process.env.CURSOR_API_KEY!,
  model: { id: "composer-2.5" },
  cloud: {
    repos: [{ url: "https://github.com/you/repo", startingRef: "main" }],
    autoCreatePR: true,
  },
  agents: { /* 各角色不同 model */ },
});

console.log(agent.agentId); // bc-... 存起来
const run = await agent.send("...");
await run.wait();
// 之后：Agent.resume(agent.agentId, { apiKey })
```

在 Cursor Web / Agents 里用 Filter → Source → **SDK** 能看到这些后台任务。

---

## 4. 和「对话界面里的模型」的关系

- 对话里选的模型 = 当前那一次聊天的模型。  
- SDK / Subagent 的 `model: { id }` = 同一账号下可用的托管模型（含多模态能力，视模型而定）。  
- **没有**官方 API 去“驱动 IDE 当前聊天窗口换模型并代你点发送”；要后台协作就用 SDK / Cloud / Automations。

---

## 5. 实用建议

1. 启动时 `Cursor.models.list()`，按角色映射模型，避免写死过期 ID。  
2. 编排用便宜/快模型，审查/难推理用强模型。  
3. 并行多 Agent 时明确分工与输出格式（JSON），最后再做一个合成步。  
4. 始终 `await run.wait()`，并用 `await using` / `with` 释放资源。  
5. 计费与 Privacy Mode 与 IDE Agent 相同，多开并行会明显烧额度。

---

如果你说一下更偏向 **纯 IDE Subagents**、**本地 Node 脚本**，还是 **Cloud 后台并行**，以及要用哪些具体模型，我可以按你的场景给一份更贴合的目录结构和 prompt 模板（仍是 Ask 模式，只给方案不直接改仓库）。需要我直接在项目里 gener 出脚手架时，切到 Agent 模式即可。

---

**User**

把上面的代码写成python版本

---

**Cursor**

我先对照 Python SDK 的官方 API，把前面的 TypeScript 示例准确翻成 Python。

下面是对应的 **Python** 版本（`pip install cursor-sdk`，Python ≥ 3.10）。

---

## 准备

```bash
export CURSOR_API_KEY="cursor_..."
pip install cursor-sdk
```

---

## 列出可用模型

```python
from cursor_sdk import Cursor

models = Cursor.models.list()  # 读 CURSOR_API_KEY
for m in models:
    print(m.id, m.display_name)
```

---

## 模式 A：父 Agent + 多模型 Subagents（自动委派）

```python
import os
from cursor_sdk import Agent, AgentDefinition, AgentOptions, LocalAgentOptions

with Agent.create(
    AgentOptions(
        api_key=os.environ["CURSOR_API_KEY"],
        model="composer-2.5",  # 编排者
        local=LocalAgentOptions(cwd=os.getcwd()),
        agents={
            "vision-analyst": AgentDefinition(
                description="看截图/UI，描述布局与缺陷",
                prompt="你是视觉分析专家。只输出结构化观察，不改代码。",
                model="gemini-3.5-flash",  # 换成 Cursor.models.list() 里的真实 ID
            ),
            "implementer": AgentDefinition(
                description="根据计划实现代码改动",
                prompt="按给定计划改代码并跑相关测试。",
                model="composer-2.5",
            ),
            "reviewer": AgentDefinition(
                description="审查改动的正确性与风险",
                prompt="只审查、列问题，不直接改。",
                model="claude-4.5-opus-high-thinking",
            ),
        },
    )
) as agent:
    result = agent.send(
        "分析这张 UI 截图，修导航栏问题，再做一次审查"
    ).wait()
    print(result.status, result.result)
```

---

## 模式 B：自己编排，并行多模型（含 VLM 看图）

并行用 **async**（官方推荐 `AsyncClient`）：

```python
import asyncio
import os

from cursor_sdk import (
    AgentOptions,
    AsyncAgent,
    AsyncClient,
    LocalAgentOptions,
    SDKImage,
    UserMessage,
)

API_KEY = os.environ["CURSOR_API_KEY"]
CWD = os.getcwd()


async def ask_model(client, model_id: str, prompt: str, image_path: str) -> str | None:
    async with await client.agents.create(
        AgentOptions(
            api_key=API_KEY,
            model=model_id,
            local=LocalAgentOptions(cwd=CWD),
        )
    ) as agent:
        run = await agent.send(
            UserMessage(
                text=prompt,
                images=[SDKImage.from_file(image_path)],
            )
        )
        result = await run.wait()
        return result.result


async def main():
    prompt = "描述图中缺陷，JSON 输出"
    image_path = "screenshot.png"

    async with await AsyncClient.launch_bridge(workspace=CWD) as client:
        a, b, c = await asyncio.gather(
            ask_model(client, "gemini-3.5-flash", prompt, image_path),
            ask_model(client, "gpt-5.5-medium", prompt, image_path),
            ask_model(client, "claude-sonnet-5-thinking-xhigh", prompt, image_path),
        )

        # 合成一步
        consensus = await AsyncAgent.prompt(
            f"综合以下三份观察，给出最终结论与修复步骤:\nA:{a}\nB:{b}\nC:{c}",
            AgentOptions(
                api_key=API_KEY,
                model="composer-2.5",
                local=LocalAgentOptions(cwd=CWD),
            ),
            client=client,
        )
        print(consensus.result)


asyncio.run(main())
```

同步版也可以，但并行要用线程池；多 Agent 编排更推荐上面的 async。

若已有 base64 字符串：

```python
UserMessage(
    text=prompt,
    images=[{"data": base64_png, "mime_type": "image/png"}],
)
# 或: SDKImage.data_image(base64_png, "image/png")
```

---

## 后台 Cloud 长期跑

```python
import os
from cursor_sdk import Agent, AgentOptions, CloudAgentOptions, CloudRepository

with Agent.create(
    AgentOptions(
        api_key=os.environ["CURSOR_API_KEY"],
        model="composer-2.5",
        cloud=CloudAgentOptions(
            repos=[
                CloudRepository(
                    url="https://github.com/you/repo",
                    starting_ref="main",
                )
            ],
            auto_create_pr=True,
        ),
        # agents={...} 可同样挂 Subagents
    )
) as agent:
    print(agent.agent_id)  # bc-... 存起来后续 resume
    result = agent.send("...").wait()
    print(result.status, result.result)

# 之后续跑：
# with Agent.resume("bc-...", AgentOptions(api_key=...)) as agent:
#     agent.send("继续...").wait()
```

---

## 和 TS 的对照

| TypeScript | Python |
|---|---|
| `model: { id: "..." }` | `model="..."` 或 `ModelSelection(id="...")` |
| `local: { cwd }` | `local=LocalAgentOptions(cwd=...)` |
| `cloud: { repos, autoCreatePR }` | `cloud=CloudAgentOptions(repos=..., auto_create_pr=True)` |
| `agents: { name: { description, prompt, model } }` | `agents={ name: AgentDefinition(...) }` |
| `await using agent` | `with Agent.create(...) as agent` |
| `await run.wait()` | `run.wait()`（sync）/ `await run.wait()`（async） |
| `Promise.all([...])` | `asyncio.gather(...)` + `AsyncClient` |

模型 ID 请用 `Cursor.models.list()` 核对，不要写死过期 ID。

---

**User**

CURSOR_API_KEY 在哪里找

---

**Cursor**

在 Cursor Dashboard 里生成：

1. 打开 [https://cursor.com/dashboard/integrations](https://cursor.com/dashboard/integrations)（需登录）
2. 找到 **API Keys**
3. 新建并复制密钥（一般形如 `cursor_...` / `crsr_...`）

团队场景也可在 **Team Settings → Service accounts** 创建服务账号密钥。

本地使用：

```bash
export CURSOR_API_KEY="粘贴你的密钥"
```

或在代码里传 `api_key=...`。密钥只显示一次，丢了就重新生成。用户密钥和团队服务账号密钥都可用于 SDK；Team Admin 类密钥目前还不支持。

