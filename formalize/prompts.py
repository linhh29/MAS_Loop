from __future__ import annotations
from .patch_types import format_operations_for_prompt

_OP_LIST = format_operations_for_prompt()

STAGE1_SYSTEM = """You are a task decomposition expert. Your job is to break down a complex task into clear sub-tasks for a multi-agent system.

Output XML with this EXACT structure:

<task_decomposition>
  <goal>one-sentence summary of the final answer to compute</goal>
  <sub_tasks>
    <sub_task idx="0">description of sub-task 0</sub_task>
    <sub_task idx="1">description of sub-task 1</sub_task>
  </sub_tasks>
</task_decomposition>

CRITICAL RULES:
- Number sub-tasks sequentially starting from idx="0"
- Each sub-task should be a self-contained unit of work
- Output 3-6 sub-tasks
- Output ONLY the XML, no markdown fences, no extra text"""

STAGE1_TEMPLATE = """Task: {task_text}

{dataset_requirement}

Break this task into sub-tasks. Each sub-task should be a clear, self-contained step that one agent can solve. Output XML only."""

STAGE2_SYSTEM = """You are a multi-agent role engineer. Given a task decomposition, assign appropriate agent roles.

Output XML with this EXACT structure:

<agent_specs>
  <agents>
    <agent id="a0" is_final="false">
      <role>detailed role description for this agent</role>
      <sub_task_idx>0</sub_task_idx>
      <dependencies>a1,a2</dependencies>
    </agent>
    <agent id="a1" is_final="true">
      <role>role for the final agent that produces the answer</role>
      <sub_task_idx>1</sub_task_idx>
      <dependencies></dependencies>
    </agent>
  </agents>
</agent_specs>

CRITICAL RULES:
- Exactly one agent must have is_final="true"
- The final agent should aggregate results and produce the final answer
- Each agent handles one sub-task (match by sub_task_idx)
- Output ONLY the XML, no markdown fences, no extra text"""

STAGE2_TEMPLATE = """Task: {task_text}

Task Decomposition:
<task_decomposition>
  <goal>{goal}</goal>
  <sub_tasks>
{sub_tasks_xml}
  </sub_tasks>
</task_decomposition>

{dataset_requirement}

Assign agent roles for each sub-task. The final agent must produce the answer in the required format. Output XML only."""

EDITOR_SYSTEM = f"""You are a MAS system optimizer. Your job is to ALWAYS improve the MAS graph, regardless of whether the current answer is correct or not.

Your output MUST be XML with this EXACT structure:

<editor_output>
  <diagnosis>
    <failure_type>category_name</failure_type>
    <evidence>what went wrong in the trace</evidence>
    <target>agent_id that needs fixing</target>
    <reasoning>why this fix should help</reasoning>
  </diagnosis>
  <operations>
    <operation op="operation_name" target="agent_id">value_text</operation>
    <operation op="operation_name" target="agent_id">value_text</operation>
  </operations>
</editor_output>

The "failure_type" should be one of the 7 improvement categories:
  - simplify: remove redundancy, merge agents
  - repair: fix errors, broken dependencies
  - upgrade: strengthen roles, add verification
  - compress: reduce token usage, early stop
  - reallocate: redistribute roles, change aggregation
  - constrain: enforce budget/agent limits
  - stabilize: add fallback, retry, consistency checks

IMPORTANT: You MUST always output at least 1 operation. NEVER output empty <operations/>.

Available operations by category:

{_OP_LIST}

CRITICAL RULES:
- ALWAYS output at least 1 <operation/>, NEVER output empty <operations/>
- If score=1.0, use failure_type "upgrade" and propose 1-2 improvements
- rewrite_role value must contain the COMPLETE new role text
- Propose 1 or more operations
- When using add_verifier: the verifier BECOMES the final agent. Its role description must instruct it to verify the upstream result and output the final \\boxed{{answer}} (correcting if wrong). Do NOT ask the verifier to only write a verification report — it must output the final answer.
- Output ONLY the XML, no markdown fences, no extra text"""

EDITOR_TEMPLATE = """Task: {task_text}
Expected Answer: {expected}

--- MAS Graph ---
{graph_xml}

--- Execution Trace ---
Actual Score: {actual_score}
Final Output: {final_output}

Agent Outputs:
{agent_traces}

Improve the MAS graph. You MUST propose at least 1 operation. If the score is 1.0 (correct), still propose improvements to make the system more robust or efficient. Output XML only."""

EDITOR_MINIMAL_TEMPLATE = """Task: {task_text}
Expected Answer: {expected}

--- MAS Graph ---
{graph_xml}

Improve the MAS graph. You MUST propose at least 1 operation. Focus on adding verification, enhancing role depth, or adding consistency checks. Output XML only."""

def get_dataset_requirement(dataset: str) -> str:
    if dataset == "aime":
        return "IMPORTANT: The final answer MUST be a single integer. Output it as \\boxed{answer}."
    elif dataset == "drb":
        return "IMPORTANT: The final answer should be a comprehensive research article with proper citations and structure."
    elif dataset == "bcp":
        return "IMPORTANT: The final answer should be a single concise answer string (e.g., a name, date, or number). Do NOT include extra explanation."
    return ""