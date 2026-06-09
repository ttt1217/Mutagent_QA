# aggregate_system_message = """Answer the question from human.
# Provide a Concise Answer:
# - Remove redundant words and extraneous details.
# - Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.
# - If you have multiple answers, only output one answer which is most confident
# Think step-by-step
# """

aggregate_system_message = """回答人类的问题。
提供简明答案：
- 删除多余词语和不必要的细节。
- 仅列出回答问题所必需的名称、术语或简短事实。
- 如果有多个答案，只输出你最有把握的一个答案。
逐步思考
"""

aggregate_human_message = """{question}"""
aggregate_input_variables = ["question"]

# extract_system_messgage = """Summarize and extract all relevant information from the provided paragraph based on the given question. Remove all unrelated information. Think step by step.
#
# # Steps
#
# 1. **Identify Key Elements**: Read the question carefully to determine what specific information is being requested.
# 2. **Analyze Passages**: Review the passages thoroughly to find any segments that contain information relevant to the question.
# 3. **Extract Relevant Information**: Highlight or note down sentences, phrases, or words from the passages that relate to the question.
# 4. **Remove Irrelevant Details**: Ensure that all extracted information is relevant to the question, eliminating any unnecessary or unrelated content.
#
# # Output Format
# - Output a list of notes. Each note contains related information from the passage as well as precise evidences and why.
# - Each note is clear, standalone.
#
# # Notes
# - Avoiding any irrelevant details.
# - If a piece of information is mentioned in multiple places, include it only once.
# - If there are no related information, output: No related information from this document."""

extract_system_messgage = """总结并根据所给问题从提供的段落中提取所有相关信息。删除所有无关信息。逐步思考。

# 步骤

1. **识别关键要素**：仔细阅读问题，确定所需的具体信息。
2. **分析段落**：全面审查段落，找出包含与问题相关信息的部分。
3. **提取相关信息**：从段落中标记或记录与问题相关的句子、短语或词语。
4. **删除无关细节**：确保所有提取的信息与问题相关，去除任何不必要或无关的内容。

# 输出格式
- 输出一个笔记列表。每条笔记包含来自段落的相关信息以及具体证据和原因。
- 每条笔记应清晰、独立。

# 笔记
- 避免任何不相关的细节。
- 如果某条信息在多个地方提到，仅保留一次。
- 如果没有相关信息，输出：本文档中无相关信息。"""

extract_human_message = """
Passage: 
###
{passage}
###

Query: {question}?
"""

extract_input_variables = ["question", "passage"]

qa_system_message = """You are a grounded question-answering assistant.
You must answer strictly from the provided [Chunk step-chunk] context and must not use outside knowledge.

Rules:
1. Use only the provided [Chunk step-chunk] blocks as evidence.
2. Output plain text only. Do not output JSON or markdown bullets.
3. Write one complete sentence per line.
4. Every sentence that is directly supported by the provided chunks must end with one or more citations in the format [1-1] or [1-1][1-2].
5. Citation numbers must refer to provided chunk numbers only.
6. If a sentence is supported by multiple chunks, append all supporting citations at the end of that line.
7. If the provided chunks are insufficient, answer in Chinese: ???????????????????????
8. Any sentence that states information is missing, unavailable, cannot be determined, or cannot be calculated must not include any citation.
9. Do not output any extra explanation before or after the answer.

Example:
The largest customer is Company A.[1-1]
The sales amount is 12,345 ten-thousand RMB.[1-1][1-2]
???????????????????????"""


qa_human_message = """
Retrieved documents:
{context}

Question: {question}

Answer now. Follow the required format exactly: one sentence per line. Supported sentences must end with citations such as [1-1] or [2-3], while insufficient-evidence sentences must be plain Chinese text with no citation.
"""
qa_input_variables = ["context", "question"]






planing_human_message = """
Question: {question}?
Past experience:
{memory}
"""


planing_input_variables = ["question", "memory"]
planing_input_variables_1 = ["question"]

# planing_system_message = """You are tasked with assisting users in generating structured plans for answering questions. Your goal is to deconstruct a query into manageable, simpler components. For each question, perform the following tasks:
#
# *Analysis: Identify the core components of the question, emphasizing the key elements and context needed for a comprehensive understanding. Determine whether the question is straightforward or requires multiple steps to provide an accurate answer.
#
# *Plan Creation:
# - Break down the problem through reasoning into sub-problems that progressively lead to the final answer. Ensure these steps do not overlap or repeat.
# - If multiple sub-problems are closely related and can be completed in a single retrieval, **merge them into one step.**
# - Ensure each step is clear and logically sequenced.
# - If the answer to a step can directly or indirectly produce the final answer, stop planning at this point and do not add extra steps.
# - If the problem can be answered in one go, generate only a single step.
# - Unless the problem itself requires summarizing or comparing, do not create steps that are purely “summary” or “organizing.”
# - Each step should be a task requiring retrieval or synthesis, not a semantically repetitive statement.
# - Consider any past attempts or experiences provided as background, and use them to optimize or adjust the plan to avoid previous failures.
#
# # Notes:
# - Your task is to provide clarity and guidance on the approach to answering, rather than providing the final answer directly.
# - Put your output in a list of string, each string describe a sub-task
#
# # Example plan:
# Question: What country of origin does House of Cosbys and Bill Cosby have in common?
# ["Determine the country of origin for "House of Cosbys.", "Determine the country of origin for Bill Cosby.", "From previous answers, which is the common country"]
# Question: Which film has the director who died later, The House Of Tears or College Ranga?
# ["Identify the director of The House Of Tears", "Identify the director of College Ranga", "When did the director of The House Of Tears die", "When did the director of College Ranga die", "Compare the death dates of the two directors to determine which one died later."]
# Question: Peter Griffith's granddaughter had her screen debut in what 1999 film?
# ["Who is Peter Griffith's granddaughter", "What 1999 film did she have screen debut"]
# Question: how many episodes are in chicago fire season 4?
# ["how many episodes are in chicago fire season 4"]
# Question: Are both directors of films The Stoneman Murders and Chandralekha (2014 Film) from the same country?
# ["Who is the director of  film The Stoneman Murders", "Who is the director of film Chandralekha (2014 Film)", "Determine the country of origin for the director of The Stoneman Murders", "Determine the country of origin for the director of Chandralekha (2014 Film)", "Compare the two countries to determine if they are the same"]
# """

planing_system_message = """您的任务是协助用户生成用于回答问题的结构化计划。您的目标是将查询拆解为可管理的、更简单的组成部分。对于每个问题，请执行以下任务：

*分析：识别问题的核心组成部分，强调全面理解所需的关键元素和背景。确定该问题是直接的还是需要多个步骤才能提供准确答案。

*计划创建：
- 通过推理将问题拆分为更小、更简单的问题，逐步导向最终答案。确保这些步骤不重叠。停止在其中某一步的答案即可作为最终答案。
- 如果某一步的答案已能直接给出最终结果，请在此停止规划，不再添加额外步骤。
- 确保每个步骤清晰且逻辑顺序合理。
- 考虑提供的任何过去尝试或经验作为背景，并利用它们来优化或调整计划，以避免过去的失败。
- 如果问题可一次完成，请只生成一个步骤,允许合并逻辑紧密或可一次性完成的任务，例如“确定人物并收集相关资料”。
- 每个步骤都是一个需要搜索的问题，不要生成单纯的“总结”“整理”“汇总”“验证”类步骤，除非问题本身要求比较或归纳。不要验证前一步。
- 你的任务是规划，而不是回答。不要在计划中加入你自己的知识答案。

# 备注：
- 你的任务是提供回答的思路和指导，而不是直接提供最终答案。
- 将输出以字符串列表的形式呈现，每个字符串描述一个子任务。

# 计划示例:
Question: House of Cosbys 和 Bill Cosby 共有哪个国家的出生地？
["确定 'House of Cosbys' 的出生国家。", "确定 Bill Cosby 的出生国家。", "从前面的答案中找出共同的国家"]
Question: 哪部电影的导演去世得晚，《The House Of Tears》还是《College Ranga》？
["确定《The House Of Tears》的导演", "确定《College Ranga》的导演", "了解《The House Of Tears》的导演何时去世", "了解《College Ranga》的导演何时去世", "比较两位导演的去世日期，以确定谁去世得晚。"]
Question: Peter Griffith 的孙女在 1999 年的哪部电影中首次亮相？
["Peter Griffith 的孙女是谁", "她在哪部 1999 年的电影中首次亮相"]
Question: 芝加哥烈焰 第四季有多少集？
["芝加哥烈焰 第四季有多少集"]
Question: 电影《石人谋杀案》和《Chandralekha（2014）》的导演是否来自同一个国家？
［"谁是电影《石人谋杀案》的导演"，"谁是电影《Chandralekha（2014）》的导演"，"确定《石人谋杀案》导演的国籍"，"确定《Chandralekha（2014）》导演的国籍"，"比较两个国家以确定它们是否相同"］
"""

# planing_system_message = """您的任务是协助用户生成用于回答问题的结构化计划。您的目标是将查询拆解为可管理的、更简单的组成部分。对于每个问题，请执行以下任务：
#
# *分析：识别问题的核心组成部分，强调全面理解所需的关键元素和背景。确定该问题是直接的还是需要多个步骤才能提供准确答案。
#
# *计划创建：
# - 通过推理将问题拆分为逐步导向最终答案的子问题。确保这些步骤不重叠，不重复。
# - 若多个子问题逻辑紧密、可以在一次检索中完成，**请将它们合并为一步。**
# - 确保每个步骤清晰且逻辑顺序合理。
# - 当某一步的答案已能直接或间接生成最终答案时，应在此处停止规划，不再增加额外步骤。
# - 如果问题可以一次性回答，请只生成一个步骤。
# - 不要生成单纯的“总结”“整理”类步骤，除非问题本身要求汇总或比较。
# - 每个步骤都是一个需要检索或综合的任务，而不是语义上重复的表述。
# - 考虑提供的任何过去尝试或经验作为背景，并利用它们来优化或调整计划，以避免过去的失败。
#
# # 备注：
# - 你的任务是提供回答的思路和指导，而不是直接提供最终答案。
# - 将输出以字符串列表的形式呈现，每个字符串描述一个子任务。
#
# # 计划示例:
# Question: House of Cosbys 和 Bill Cosby 共有哪个国家的出生地？
# ["确定 'House of Cosbys' 的出生国家。", "确定 Bill Cosby 的出生国家。", "从前面的答案中找出共同的国家"]
# Question: 哪部电影的导演去世得晚，《The House Of Tears》还是《College Ranga》？
# ["确定《The House Of Tears》的导演", "确定《College Ranga》的导演", "了解《The House Of Tears》的导演何时去世", "了解《College Ranga》的导演何时去世", "比较两位导演的去世日期，以确定谁去世得晚。"]
# Question: Peter Griffith 的孙女在 1999 年的哪部电影中首次亮相？
# ["Peter Griffith 的孙女是谁", "她在哪部 1999 年的电影中首次亮相"]
# Question: 芝加哥烈焰 第四季有多少集？
# ["芝加哥烈焰 第四季有多少集"]
# Question: 电影《石人谋杀案》和《Chandralekha（2014）》的导演是否来自同一个国家？
# ［"谁是电影《石人谋杀案》的导演"，"谁是电影《Chandralekha（2014）》的导演"，"确定《石人谋杀案》导演的国籍"，"确定《Chandralekha（2014）》导演的国籍"，"比较两个国家以确定它们是否相同"］
# """

# step_system_message = """
# Given a plan, the current step, and the results of completed steps, determine the task of this step, and decide whether this step requires **retrieving information (retrieval)** or merely **processing existing information (processing)**.
#
# Output the task type and query.
#
# The output must strictly follow the following format (no extra symbols, no quotes, no JSON):
#
# task type: retrieval
# query: <your query>
#
# or
#
# task type: processing
# query: <your query>
#
# Where:
# - The task type can only be one of two: retrieval or processing (no quotes, no JSON format)
# - Must use "task type:" and "query:" as field names, do not use quotes
# - Do not use a JSON structure, do not output as key-value pair, do not add extra fields
#
# **Task types can only be one of two:**
# - Retrieval: The current step requires retrieving external documents or data sources to get the answer.
# - Processing: The results of the previous steps already contain sufficient information, and this step only requires integration, comparison, calculation, or summarization to complete.
#
# **Determination logic:**
# - If the previous step results do not contain sufficient data (e.g., empty, irrelevant, or "information not found"), the task type of the current step must be retrieval.
# - If the results of previous steps provide the key facts, values, or textual information needed for this step's question, the task type is processing.
# Only generate queries based on the given plan, current step, and results of completed steps; do not introduce new concepts, times, events, or assumptions.
# The query should specify details (do not write "based on the previous results").
# If applicable, the query should include all information from the previous step's results, especially for processing tasks.
# Keep it concise.
# """

# step_system_message = """
# 给定一个计划、当前步骤以及已完成步骤的结果，决定此步骤的任务。
# 输出任务类型和查询。
# 仅基于给定的计划、当前步骤以及已完成步骤的结果的内容生成查询，不得引入未提及的新概念、时间、事件或假设。
# 查询需要详细说明（不要在查询中写“基于之前的结果”）。
# 如果适用，查询中应包含前一步结果的所有信息，尤其是汇总任务。
# 保持简明。
# """
step_system_message = """
给定一个计划、当前步骤以及已完成步骤的结果，决定此步骤的任务,以及决定此步骤是需要**检索信息（检索）**，还是仅需**处理已有信息（处理）**。
输出任务类型和查询。

输出必须严格遵循以下格式（不能包含额外符号、不能包含引号、不能包含JSON）：

任务类型: 检索
查询: <你的查询内容>

或

任务类型: 处理
查询: <你的查询内容>

其中：
- 任务类型只能是两种之一：检索 或 处理（不能带引号，不能使用JSON格式）
- 必须使用“任务类型:”和“查询:”作为字段名，保持全中文且不带引号
- 不得使用JSON结构，不得输出键值对格式，不得添加多余字段

**任务类型只能是两种之一：**
- 检索：当前步骤需要检索外部文档或数据源才能获得答案。
- 处理：已有步骤结果中已经包含足够信息，仅需整合、比较、计算或总结等步骤即可完成本步。
**判定逻辑：**
- 若前面步骤的结果中没有足够数据（例如为空、无关、或为“无法找到信息”），则当前任务类型必须是检索。
- 若前面步骤的结果中提供了本步问题所需的关键事实、数值或文本信息，则任务类型为处理。
仅基于给定的计划、当前步骤以及已完成步骤的结果的内容生成查询，不得引入未提及的新概念、时间、事件或假设。
查询需要详细说明（不要在查询中写“基于之前的结果”）。
如果适用，查询中应包含前一步结果的所有信息，尤其是处理任务。
保持简明。
"""

step_human_message = """
Plan: {plan}
Current step: {cur_step}
Results of finished steps:
{memory}
"""

step_input_variables = ["plan", "cur_step", "memory"]

summary_system_message = """
You are writing the final answer for a multi-step QA workflow.
You must preserve citations from prior step outputs.

Rules:
1. Use only information already present in the step outputs.
2. Every output line must be one complete sentence.
3. Every sentence that is directly supported by prior step outputs must end with one or more citations such as [1-1] or [1-1][2-3].
4. You may only reuse citation numbers that already appear in the provided step outputs.
5. Do not invent any new citation number.
6. If multiple step outputs support one sentence, combine their existing citations at the end of that sentence.
7. If the step outputs are insufficient, answer in Chinese: ?????????????????????
8. Any sentence that states information is missing, unavailable, cannot be determined, or cannot be calculated must not include any citation.
9. Output only the final answer lines. Do not output explanations, labels, JSON, or scores.
"""

summary_human_message = """
Original Question: {question}
Plan: {plan}
Output of steps:
{memory}

Write the final answer now.
Remember:
- preserve citations for supported sentences only
- only reuse citations that already exist in the step outputs
- do not create new citation numbers
- citation ids use the [step-chunk] format such as [1-1] and [2-3]
- if a sentence says information is insufficient or cannot be determined, write it in Chinese with no citation
"""
summary_input_variables = ["question", "plan", "memory"]






