# Git 提交信息生成规则（全局）

1. 语言强制：提交信息必须使用中文（简体优先）。
2. 语言例外：仅允许英文/ASCII 出现在以下位置：
   - Conventional Commits 的 type / scope / BREAKING CHANGE
   - 代码符号、标识符、文件名、命令、协议名、版本号、路径
3. 禁止英文叙述：除第2条例外，subject 与 body 禁止整句英文表述。

4. 格式强制（Conventional Commits）：
   - 第一行必须为：<type>(可选 scope): <subject>
   - 允许的 type：feat, fix, docs, style, refactor, perf, test, chore, ci, build
   - scope：只有在非常明确且能稳定复用时才写；不确定就省略。

5. subject（中文）规则：
   - 必须以中文动作动词开头（例如：添加/修复/优化/重构/更新/移除/纠正/禁用）
   - 不要以句号结尾，不要加“本次/这里/一些”等无信息词
   - 长度：优先 ≤ 50 字符；必要时允许到 72 字符，但要尽量精炼

6. body（可选，中文）规则：
   - subject 后必须空一行再写 body
   - 用 1–3 条要点说明“做了什么 + 为什么”（避免展开实现细节/代码过程）
   - 每行建议在 ~72 字符附近换行
   - 要点格式固定为以“- ”开头

7. 破坏性变更（Breaking Change）强制标记：
   - 只要可能导致现有调用方/使用方不兼容，必须标记 breaking change
   - 标记方式二选一：
     a) header 用 !：type(scope)!: <subject>
     b) footer 加一行：BREAKING CHANGE: <中文说明（包含影响面与迁移建议）>

8. footer（可选）：
   - 可在 body 后空一行添加元信息行，例如：Refs: <编号> / Fixes: <编号>
   - 不要在 footer 写长段落；长说明放 body

9. 输出限制：
   - 只输出最终提交信息（不要解释、不要引号、不要前后缀文本）
   - 不要提到 AI/Copilot/模型/生成等字样

10. 准确性与责任：
   - 提交信息必须与实际 diff 一致；生成后必须人工复核并按需修改

【输出模板（严格遵守）】
<type>(可选 scope): <中文动词开头的 subject>

- <要点1：做了什么 + 为什么>
- <要点2（可选）>
- <要点3（可选）>

BREAKING CHANGE: <仅在破坏性变更时填写，中文说明 + 迁移建议>
Refs: <可选>
