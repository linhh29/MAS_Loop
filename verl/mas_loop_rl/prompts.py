ORCHESTRA_SYSTEM = """You are Orchestra, a local MAS generator. Design a multi-agent system, not the final answer. Output ONLY one valid XML <mas> document. Use <agents><agent ...><role>...</role><tools>...</tools></agent></agents> and <edges><edge ... /></edges>. Set id, budget, and execution_mode on every agent. execution_mode must be llm_only or multi_turn_search. Use 2-8 agents. Put topology, aggregation, max_cost, and max_iterations on the <mas> root. Escape XML special characters in text. No JSON, markdown, <think>, or private reasoning. Stop immediately after </mas>."""
BCP_ORCHESTRA_SYSTEM = """You are MAS-Loop's Orchestra policy. Design a multi-agent system, not the final answer. Return exactly one compact XML <mas> document with no JSON, markdown, or private reasoning. Do not solve the problem directly. Stop after </mas>."""
AIME_ORCHESTRA_SYSTEM = """You are Orchestra for AIME mathematical problem solving. Design a multi-agent system, not the answer. Output ONLY one valid XML <mas> document. Use exactly 5 llm_only agents with empty tools (<tools></tools>); never invent retrieval, symbolic_math, brute_force, calculator, or other tools. Use the fixed repair-and-verify DAG described in the user prompt: two independent solver drafts in parallel, one forensic error-localizer that repairs them, one clean-room verifier that receives only the repaired solution, and one final adjudicator that receives only the repaired solution and clean-room verification. Raw draft outputs must not flow directly to the final adjudicator, because repeated unsupported answers create anchoring rather than evidence. Every node must work from the original problem and give exact mathematics. Require explicit checks for domains, all simultaneous coordinate bounds, endpoints, zero/sign cases, overlaps, duplicate counts, completeness of finite cases, factor expansions/substitution, modular conditions, probability sample spaces, geometric constraints, and requested rounding or remainder. A downstream node must restart the decisive calculation when all upstream support is invalid; it must never compromise between numbers or select a majority answer. The final output must be exactly one AIME integer in \boxed{...}. Set valid ids, budgets, execution_mode, topology, aggregation, max_cost, max_iterations, and valid data_flow edges. No JSON, markdown, <think>, or private reasoning. Stop immediately after </mas>."""
GPQA_ORCHESTRA_SYSTEM = """You are Orchestra for GPQA Diamond scientific multiple-choice problem solving. Design a multi-agent system, not the answer. Output ONLY one valid XML <mas> document. Use exactly 3 llm_only agents with empty tools (<tools></tools>); never invent browsing, BM25, databases, calculators, simulators, or domain tools. Use exactly the fixed DAG levels [2,1]: one independent derivation specialist and one independent option-falsification specialist in parallel, both feeding one reviewer/final agent. Every agent must solve from the original prompt rather than confidence, verbosity, majority, or familiarity. A calculation must explicitly separate mantissa and power-of-ten arithmetic, carry units, and perform a dimension/order-of-magnitude check. A reaction or spectroscopy question must track the carbon skeleton, functional groups, regiochemistry/stereochemistry, and each intermediate rather than saying likely. A conceptual or biological question must test the exact quantifier and definition against every option. If an apparent calculation makes all options false, recheck units, signs, exponents, assumptions, and option interpretation; never select a knowingly false closest option. The reviewer must inspect the shared premise even when specialists agree. When they disagree, it must reconstruct the single deciding fact/calculation itself and compare both leading options explicitly. Set valid ids, budgets, execution_mode, topology, aggregation, max_cost, max_iterations, and data_flow edges. The final agent must end with exactly one uppercase choice in \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}. No JSON, markdown, <think>, or private reasoning. Stop immediately after </mas>."""
AIME_EDITOR_SYSTEM = """You are Refiner for AIME. Be exceptionally conservative because editing often damages a correct exact answer. Prefer <operation op=\"accept\" />. Rewrite exactly one existing role only when the visible trajectory contains a concrete, attributable algebraic, arithmetic, boundary, duplicate-count, or required-format error; uncertainty, disagreement, a missing telemetry field, or a merely suspicious answer is not evidence. Never use gold, reward, correctness labels, or benchmark outcomes. A valid repair must preserve the graph and add no fake tools or redundant agents, and must state a falsifiable benefit. Output exactly one grounded XML <refiner> document with the required three stages and exactly one operation."""
ALFWORLD_EDITOR_SYSTEM = """You are Refiner for ALFWorld. Make one task-blind prospective capability repair to the supplied baseline MAS. Missing legacy telemetry is unknown, not runtime failure. Preserve the graph, agents, tools, budgets, edges, and final aggregator. Rewrite exactly one generic NON-AGGREGATION solver, normally solver_0, and ground the rewrite only with that solver's node_cards[i].declared_responsibility location. Never rewrite the aggregation agent. The reusable replacement must be a live-state progress specialist: use the CURRENT task and observation, inventory, recent action-observation history, goal predicates and prerequisites, and exact admissible actions; compare executable candidates by whether they create a required state, preserve completed progress, avoid premature placement or undoing useful state, and reduce unnecessary navigation or inspection. Give one concise state-grounded reason, then end with exactly one executable action copied verbatim from the admissible list as Action: <action>. Never copy instance objects, receptacles, locations, numbers, or initial-state wording into the role. Never use outcome fields, reward signals, hidden answers, or benchmark labels. Output exactly one compact XML <refiner> document with the required three stages and exactly one rewrite_role operation; no JSON, markdown, or private reasoning."""
GPQA_EDITOR_SYSTEM = """You are Refiner for GPQA Diamond. Be conservative and task-blind. Missing telemetry is unknown, never a runtime failure. Preserve graph, agents, tools, and budgets. You may rewrite exactly one existing duplicated/generic role when source_mas itself visibly lacks complementary scientific verification; this is a prospective capability repair and does not require execution telemetry. Prefer rewriting one generic solver into a reusable option-by-option scientific falsifier that uses prompt facts and governing laws/equations, separates assumptions, checks units/sign/scale, and compares the strongest alternative. Never copy task-specific entities, numbers, reactions, answer content, or gold into a role. Do not add a verifier agent or make a format-only verifier. Otherwise accept. Never use reward, correctness labels, or benchmark outcomes. Output exactly one grounded XML <refiner> document with the required three stages and exactly one operation."""
EDITOR_SYSTEM = """You are Refiner, a conservative dataset-neutral MAS diagnosis-and-edit policy. Inspect the supplied source MAS, public task, imported execution trajectory, and bounded diagnostics. Prefer preserving a functioning source: emit accept unless one concrete observed signal supports one causal, non-redundant edit. IMPORTANT: imported legacy trajectories may not contain per-agent telemetry. A diagnostic execution_status=missing, missing_execution_trace, upstream_missing, aggregation_not_executed, or empty_final_output caused only by absent trace fields means UNKNOWN TELEMETRY, not that an agent failed to run; never repair it by adding an agent, adding an already-present edge, or increasing budget. The existing graph is executable and direct solver-to-aggregator edges already deliver outputs. Valid evidence is an actual visible model output, retrieval record, simulator action/observation, parser error, repeated/stalled action, format violation, or an explicit role/capability mismatch visible in source_mas. A duplicated generic role may justify rewriting exactly one existing role into a complementary task-specific solver/verifier, but do not add graph bloat. For live interactive tasks, any rewritten role must reason from the CURRENT state and exact admissible-action list, never memorize the initial state. First localize the earliest evidence-supported failure or capability gap, then state a falsifiable counterfactual, then emit exactly one operation: accept or one bounded patch operation. Never use hidden gold answers, correctness labels, rewards, or benchmark outcome fields. Never emit multiple operations. Allowed decisions: accept, add_agent, rewrite_role, add_edge, remove_edge, change_budget, replace_tool. Reject redundant edges and generic new agents. accept must be the only operation when evidence is insufficient. Output exactly one XML <refiner> document containing stage1_localization, stage2_causal_model, stage3_repair_plan, and patch. Escape XML special characters. No JSON, markdown, <think>, or private reasoning. Stop immediately after </refiner>."""
SUBAGENT_SYSTEM = "You are a local sub-agent inside a multi-agent system. Follow your role, use prior context, and produce a concise useful answer. Do not output <think> or private reasoning; give only the useful contribution and final answer when applicable."


def orchestra_prompt(task: str, constraints: dict | None = None) -> list[dict[str, str]]:
    constraints = constraints or {}
    dataset = str(constraints.get("dataset") or constraints.get("evaluation_dataset") or "").strip().lower()
    try:
        max_agents = max(2, min(6, int(constraints.get("max_agents", 6))))
    except (TypeError, ValueError):
        max_agents = 6
    # Use explicit open/close tags for empty containers.  The base model often
    # copied a self-closing <tools /> and then hallucinated a second </tools>,
    # which made otherwise-correct MAS outputs invalid XML.
    retrieval_example = '''<mas topology="dag" aggregation="agent_2" max_cost="3" max_iterations="3"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Acquire task-specific evidence.</role><tools><tool>bm25</tool></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Reconcile upstream evidence.</role><tools></tools></agent><agent id="agent_2" budget="1" execution_mode="llm_only"><role>Return the exact final answer.</role><tools></tools></agent></agents><edges><edge from="agent_0" to="agent_1" type="data_flow" /><edge from="agent_1" to="agent_2" type="data_flow" /></edges></mas>'''
    aime_example = r'''<mas topology="dag" aggregation="agent_4" max_cost="5" max_iterations="5"><agents><agent id="agent_0" budget="1" execution_mode="llm_only"><role>Formal primary solver: translate every sentence of the original problem into explicit mathematical constraints, classify the problem, and derive an exact answer without relying on undocumented computation. For counting construct a recurrence, bijection, inclusion-exclusion, or complete case table; for algebra and number theory track domains, integrality, signs, valuations, and substitutions; for geometry derive all claimed lengths or angles. Check the result against the original conditions and end with one boxed integer.</role><tools></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Independent alternate-method solver: ignore any other draft and solve the original problem from scratch using a genuinely different representation. Stress-test endpoints, zero and sign cases, all simultaneous variable bounds, overlaps, duplicate counting, small cases, modular residues, diagram assumptions, and requested rounding or remainder. Do not guess a standard result; expose the decisive computation and end with one boxed integer.</role><tools></tools></agent><agent id="agent_2" budget="1" execution_mode="llm_only"><role>Forensic error-localizer and repair solver: inspect both drafts claim by claim against the original problem. Identify the earliest invalid or unsupported decisive step in each draft, including a shared premise even when their answers agree. Then construct one corrected solution yourself; do not average, compromise, or vote. If neither draft is sound, restart from the formal constraints. Explicitly verify finite-case completeness, arrangement multipliers, simultaneous bounds, factor expansions or substitutions, and arithmetic. End with the repaired candidate in one boxed integer.</role><tools></tools></agent><agent id="agent_3" budget="1" execution_mode="llm_only"><role>Clean-room certificate verifier: receive only the repaired solution, not the noisy raw drafts. Recompute the shortest decisive invariant from the original problem by an independent check. Try actively to falsify the repaired answer using boundary cases, reverse substitution, an alternate count, a small-instance test, modular or valuation checks, or coordinate verification as appropriate. If it fails, produce the corrected derivation rather than merely criticizing it. End with the uniquely certified boxed integer.</role><tools></tools></agent><agent id="agent_4" budget="1" execution_mode="llm_only"><role>Final adjudicator: use only the repaired solution and its clean-room verification together with the original problem. Validate the exact step that determines the numerical answer. Treat agreement as no evidence unless the certificate is reproducible; if the two upstream results differ, redo that separating calculation and choose the mathematically valid result, never a majority or numerical compromise. Preserve leading zeros conceptually and output exactly one AIME integer in \boxed{...} as the final nonempty line.</role><tools></tools></agent></agents><edges><edge from="agent_0" to="agent_2" type="data_flow" /><edge from="agent_1" to="agent_2" type="data_flow" /><edge from="agent_2" to="agent_3" type="data_flow" /><edge from="agent_2" to="agent_4" type="data_flow" /><edge from="agent_3" to="agent_4" type="data_flow" /></edges></mas>'''
    gpqa_example = r'''<mas topology="dag" aggregation="agent_2" max_cost="3" max_iterations="3"><agents><agent id="agent_0" budget="1" execution_mode="llm_only"><role>Independent derivation specialist: solve the original question from first principles and commit to one option. State the one definition, law, mechanism, structural invariant, or experimental implication that decides it. For numerical physics or chemistry, write SI conversions with mantissa and exponent separately, propagate units, and sanity-check dimensions and order of magnitude. For reactions or spectra, name each intermediate and track carbon connectivity, functional groups, regiochemistry, stereochemistry, and signals step by step. For biology or conceptual questions, honor exact qualifiers such as except, always, and necessary. Explicitly test the strongest distractor and end with one boxed uppercase option.</role><tools></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Independent option-falsification specialist: solve without seeing another attempt. Build a compact option-by-option elimination based only on prompt facts and established science, then reproduce the shortest decisive calculation or mechanism. Separate stated facts from assumptions. Check unit powers, signs, scale, limiting cases, molecular formula and skeleton conservation, reaction feasibility, stereochemistry, and exact biological definitions as applicable. If every option appears wrong, treat that as evidence of an error in your premise or arithmetic and restart; never choose a knowingly false closest option. End with one boxed uppercase option.</role><tools></tools></agent><agent id="agent_2" budget="1" execution_mode="llm_only"><role>Main reviewer and final answerer: independently adjudicate the two attempts against the original question. First extract each specialist's decisive premise, not its confidence or answer letter. Reproduce the single discriminating check yourself. For calculations, explicitly recompute powers of ten and units and perform a scale check; for reaction routes or NMR, reconstruct every intermediate and carbon skeleton; for conceptual science, test the exact definition and quantifier against the two leading options. Agreement may reflect a shared unsupported premise, so verify it once. Under disagreement, compare both leading options symmetrically and reject one only with a concrete contradiction. If your reasoning says all options are false, locate and repair the mistaken premise instead of selecting the closest. Never vote or reward detail. End with exactly one of \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}.</role><tools></tools></agent></agents><edges><edge from="agent_0" to="agent_2" type="data_flow" /><edge from="agent_1" to="agent_2" type="data_flow" /></edges></mas>'''
    if dataset in {"bcp", "browsecomp", "browsecomp_plus", "browsecomp-plus"}:
        user = f"""Dataset context: bcp
Task:
{task}

Output one XML document shaped like:
{retrieval_example}

Constraints: Use 3-{max_agents} agents and decompose the task into 3-{max_agents} clear self-contained sub-tasks. Give every agent a task-specific role tied to the current clues; Generic or reusable placeholder roles are invalid. Include exactly one aggregation/final agent, set execution_mode on every agent, and edges must reference earlier agent ids and all edge endpoints must be valid. Use multi_turn_search only when an agent must acquire NEW facts/documents from local BM25; use llm_only for reasoning over task context and upstream evidence. Include at least one multi_turn_search evidence agent. Do not make every agent a retrieval agent. The aggregation agent should normally be llm_only, synthesize only from retrieved upstream snippets, and return the exact short answer. Set tools=["bm25"] represented as <tools><tool>bm25</tool></tools> for search agents and tools=[] represented as <tools></tools> for llm_only agents. Always close every <agent> with </agent>. Never add a closing tag after a self-closing tag. Set budget=1 for every agent and max_cost to the sum of budgets. Return XML only."""
        return [{"role": "system", "content": BCP_ORCHESTRA_SYSTEM}, {"role": "user", "content": user}]
    if dataset == "aime":
        system = AIME_ORCHESTRA_SYSTEM
        guidance = (
            "Use exactly 5 complementary llm_only agents with empty tools in the fixed repair-and-verify DAG: "
            "two independent drafts in parallel, one error-localizer/repair solver, one clean-room verifier that sees "
            "only the repaired solution, and one final adjudicator that sees only the repair and verification. Do not "
            "send noisy raw drafts directly to the final node. No retrieval or BM25 is available."
        )
        example = aime_example
    elif dataset in {"gpqa", "gpqa_diamond", "gpqa-diamond"}:
        system = GPQA_ORCHESTRA_SYSTEM
        guidance = (
            "Use exactly 3 complementary llm_only agents with empty tools in levels [2,1]: one independent derivation "
            "specialist and one independent option-falsification specialist, followed directly by one reviewer/final answerer. "
            "The reviewer must reproduce the deciding fact, calculation, or mechanism, check shared premises even under "
            "agreement, and never choose a knowingly false closest option. No retrieval or BM25 is available."
        )
        example = gpqa_example
    else:
        system = ORCHESTRA_SYSTEM
        guidance = f"Use 3-{max_agents} agents and task-specific roles."
        example = retrieval_example
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task:\n{task}\n\nConstraints:\n{constraints}\n\n{guidance} Always close every <agent> with </agent>, and never add a closing tag after a self-closing tag. Example structure:\n{example}"},
    ]


def compact_trajectory_for_editor(src_trajectory: dict, src_mas: dict | None = None, task: str = "", constraints: dict | None = None) -> dict:
    """Compatibility wrapper returning the bounded diagnostic state."""
    from mas_loop_rl.diagnostics import build_node_diagnostic_state
    if src_mas is None:
        src_mas = src_trajectory.get("mas_config", {}) if isinstance(src_trajectory, dict) else {}
    return build_node_diagnostic_state(src_mas or {}, src_trajectory or {}, task, constraints)


_EDITOR_CANDIDATE_FOCI = (
    "preserve_or_accept: test whether the source should be preserved; accept only when no evidence-backed edit is expected to help",
    "earliest_node_failure: target the earliest agent whose responsibility, received inputs, or output shows an evidence-backed causal failure",
    "dataflow_failure: inspect an edge or handoff for missing, misrouted, or unusable information and repair the delivery path",
    "aggregation_failure: inspect whether the final/aggregation node used upstream contributions and fulfilled exact final-output responsibility",
    "capability_failure: inspect role, execution mode, tools, or budget for a concrete mismatch with the assigned responsibility",
    "alternative_node_hypothesis: choose a plausible target different from the most obvious generic planner/final-node diagnosis",
    "minimal_causal_repair: make exactly one bounded operation that directly fixes one supported causal mechanism",
    "single_edit_repair: choose the one highest-value bounded operation for the diagnosis and repair goal",
)


def editor_candidate_focus(candidate_index: int, candidate_count: int) -> str:
    base = _EDITOR_CANDIDATE_FOCI[candidate_index % len(_EDITOR_CANDIDATE_FOCI)]
    return f"candidate {candidate_index + 1}/{candidate_count}; {base}"


def _editor_agent_ids(src_mas: dict) -> list[str]:
    agents = src_mas.get("agents") if isinstance(src_mas, dict) else []
    if not isinstance(agents, list):
        return []
    return [str(agent.get("id")) for agent in agents if isinstance(agent, dict) and str(agent.get("id") or "").strip()]


def editor_operation_contract(src_mas: dict) -> str:
    ids = _editor_agent_ids(src_mas)
    known = ",".join(ids) or "<none>"
    return (
        f"Existing agent IDs: [{known}]. Use these exact IDs when possible. "
        'Patch operation XML forms: <operation op="accept" />; '
        '<operation op="rewrite_role" agent_id="ID"><role>NEW_ROLE</role></operation>; '
        '<operation op="change_budget" agent_id="ID" budget="1" />; '
        '<operation op="replace_tool" agent_id="ID" old_tool="OPTIONAL" new_tool="TOOL" />; '
        '<operation op="add_agent" id="NEW_ID" budget="1"><role>ROLE</role></operation>; '
        '<operation op="add_edge" from="ID" to="OTHER_ID" type="data_flow" />; '
        '<operation op="remove_edge" from="ID" to="OTHER_ID" type="data_flow" />. '
        "Emit exactly one operation. The parser retains legacy aliases only for audit/rejection; do not use them. budget is 1-4."

    )


def editor_retry_instruction(src_mas: dict) -> str:
    return (
        "Repair the previous output as one complete XML <refiner> document. Preserve only diagnosis that is "
        "grounded in node_diagnostic_state. Every evidence item must use a location copied exactly from a "
        "trace_refs value and must provide matching agent_id, execution_index, optional retrieval round, and "
        "field. Every <item> must contain non-empty task-specific observation text between its opening and "
        "closing tags. Write what that exact referenced trace value concretely shows; do not use schema labels, "
        "generic placeholder phrases, or an empty item body as evidence. Fix every field named by the validation error, "
        "emit no JSON/markdown, and stop after </refiner>. "
        + editor_operation_contract(src_mas)
    )


def _alfworld_editor_prompt(src_mas: dict, diagnostic_state: dict) -> list[dict[str, str]]:
    """Compact, label-blind prompt for prospective ALFWorld role repair.

    Legacy ALFWorld rows contain large parser-error trajectories but no useful
    per-step agent telemetry. Sending that material to the editor consumed most
    of its context window and caused malformed/truncated patches. The source
    role itself is sufficient grounding for this bounded capability rewrite.
    """
    import json

    agents = []
    node_evidence = []
    cards = diagnostic_state.get("node_cards") if isinstance(diagnostic_state, dict) else []
    cards_by_id = {
        str(card.get("agent_id")): card for card in (cards or []) if isinstance(card, dict)
    }
    for index, agent in enumerate(src_mas.get("agents") or []):
        if not isinstance(agent, dict):
            continue
        aid = str(agent.get("id") or "")
        agents.append({
            "id": aid,
            "role": str(agent.get("role") or ""),
            "is_aggregation": aid == str(src_mas.get("aggregation") or ""),
        })
        card = cards_by_id.get(aid, {})
        trace_refs = card.get("trace_refs") if isinstance(card, dict) else {}
        node_evidence.append({
            "agent_id": aid,
            "execution_index": card.get("execution_index", index),
            "declared_responsibility": card.get("declared_responsibility", agent.get("role", "")),
            "is_aggregation": aid == str(src_mas.get("aggregation") or ""),
            "role_location": (trace_refs or {}).get(
                "declared_responsibility", f"node_cards[{index}].declared_responsibility"
            ),
        })
    payload = {
        "source_mas": {
            "agents": agents,
            "edges": src_mas.get("edges") or [],
            "aggregation": src_mas.get("aggregation"),
            "topology": src_mas.get("topology"),
            "max_cost": src_mas.get("max_cost"),
            "max_iterations": src_mas.get("max_iterations"),
        },
        "node_role_evidence": node_evidence,
        "public_execution_contract": (
            "At every environment step the executor supplies the current task, observation, inventory, "
            "recent action-observation history, goal guidance, and exact admissible actions."
        ),
    }
    shape = (
        '<refiner><stage1_localization source_status="actionable_failure" target_type="agent" '
        'target_id="SOLVER_ID" failure_type="generic_role_capability_gap"><evidence>'
        '<item agent_id="SOLVER_ID" execution_index="INDEX" field="declared_responsibility" '
        'location="ROLE_LOCATION">The declared generic solver role does not track live goal predicates, '
        'prerequisites, or completed progress.</item></evidence></stage1_localization>'
        '<stage2_causal_model><causal_chain><step>A generic candidate can ignore live state progress.</step>'
        '</causal_chain><causal_explanation>The missing progress contract weakens next-action comparison.'
        '</causal_explanation><counterfactual>A live-state progress specialist supplies a grounded candidate and reason.'
        '</counterfactual><downstream_impact>The unchanged aggregator receives a more useful independent candidate.'
        '</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="0.9"><repair_goal>Rewrite one '
        'generic solver only.</repair_goal><expected_effect>Improve progress-preserving next-action selection.'
        '</expected_effect><risk>The specialist may over-prioritize short-term progress.</risk></stage3_repair_plan><patch>'
        '<operation op="rewrite_role" agent_id="SOLVER_ID"><role>Live-state progress specialist: From the CURRENT task and observation, inventory, recent action-observation history, exact admissible actions, and goal predicates with their prerequisites, compare only executable candidates by whether they create a required state and preserve completed progress. Avoid premature placement, unnecessary navigation or inspection, and undoing useful state. Give one concise state-grounded reason, then end with exactly one executable action copied verbatim from the admissible action list in Action: one admissible action form.</role></operation>'
        '</patch></refiner>'
    )
    user = (
        "Return one compact XML document only. Select one non-aggregation generic solver (prefer solver_0). "
        "Copy its agent_id, execution_index, and role_location exactly from node_role_evidence into the evidence item. "
        "Replace only SOLVER_ID, INDEX, and ROLE_LOCATION in this shape; do not add operations or alter the graph. "
        "Copy the complete role text in the shape exactly as written. It is a reusable instruction, not a request to invent "
        "an environment action now; never replace its final generic wording with an example action or placeholder.\n"
        + shape + "\nInput JSON (source architecture and role evidence only):\n" + json.dumps(payload, ensure_ascii=False)
    )
    return [{"role": "system", "content": ALFWORLD_EDITOR_SYSTEM}, {"role": "user", "content": user}]


def editor_prompt(task: str, src_mas: dict, src_trajectory: dict, constraints: dict | None = None, *, candidate_index: int = 0, candidate_count: int = 1, repair_focus: str | None = None) -> list[dict[str, str]]:
    import json
    from mas_loop_rl.diagnostics import sanitize_refiner_constraints
    dataset = str((constraints or {}).get("dataset") or (constraints or {}).get("evaluation_dataset") or "").strip().lower()
    diagnostic_state = compact_trajectory_for_editor(src_trajectory, src_mas, task, constraints)
    if dataset in {"alfworld", "alfw"}:
        return _alfworld_editor_prompt(src_mas, diagnostic_state)
    payload = {
        "task": task,
        "constraints_and_capabilities": sanitize_refiner_constraints(constraints or {}),
        "source_mas": src_mas,
        "node_diagnostic_state": diagnostic_state,
    }
    if repair_focus:
        payload["optional_sampling_focus"] = repair_focus
    evidence_contract = (
        "Each <evidence><item> must have attributes agent_id, execution_index, field, and location; add round "
        "when the location references a retrieval round. location must be copied exactly from one trace_refs "
        "value in node_diagnostic_state. The item text must state the concrete observed signal at that location. "
        "For an edge use agent_id='from->to' and execution_index='-1'. For a MAS/outcome observation use "
        "agent_id='mas' and execution_index='-1'. Do not copy field descriptions as observations."
    )
    shape = (
        '<refiner><stage1_localization source_status="..." target_type="..." target_id="..." failure_type="...">'
        '<evidence><item agent_id="..." execution_index="..." field="..." location="..." round="...">'
        'Concrete task-specific observation from this exact trace value; replace this example.'
        '</item></evidence></stage1_localization><stage2_causal_model><causal_chain><step></step></causal_chain>'
        '<causal_explanation></causal_explanation><counterfactual></counterfactual><downstream_impact>'
        '</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="..."><repair_goal>'
        '</repair_goal><expected_effect></expected_effect><risk></risk></stage3_repair_plan><patch>'
        '<operation op="..." /></patch></refiner>'
    )
    user = (
        "Produce one conservative grounded three-stage decision. Missing legacy telemetry is not failure. Before editing, verify that the proposed operation is novel in source_mas and directly addresses an observed output/action/capability signal; otherwise accept. Do not add agents merely because node execution traces are unavailable. "
        "source_status must be exactly one of: actionable_failure, preserve; target_type must be agent, edge, aggregation, or mas. "
        "Use this XML element/attribute contract. Replace every ellipsis/example, fill all elements with "
        "task-specific content, and never emit an empty <evidence><item></item> body:\n" + shape + "\n" + evidence_contract + "\n" + editor_operation_contract(src_mas)
        + "\n\nInput state is JSON data embedded only as input context; your OUTPUT must be XML:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    if dataset == "aime":
        editor_system = AIME_EDITOR_SYSTEM + " " + EDITOR_SYSTEM
    elif dataset in {"gpqa", "gpqa_diamond", "gpqa-diamond"}:
        editor_system = GPQA_EDITOR_SYSTEM + " " + EDITOR_SYSTEM
    else:
        editor_system = EDITOR_SYSTEM
    return [{"role": "system", "content": editor_system}, {"role": "user", "content": user}]


def orchestra_revision_prompt(task: str, previous_mas: dict, editor_patch: dict, updated_trajectory: dict, round_index: int, constraints: dict | None = None) -> list[dict[str, str]]:
    """Prompt Orchestra for the next full MAS in one stateful loop trajectory."""
    import json
    payload = {
        "task": task,
        "loop_round": round_index,
        "previous_updated_mas": previous_mas,
        "previous_editor_patch": editor_patch,
        "previous_updated_trajectory": updated_trajectory,
        "constraints": constraints or {},
    }
    dataset = str((constraints or {}).get("dataset") or (constraints or {}).get("evaluation_dataset") or "").strip().lower()
    if dataset == "aime":
        system = AIME_ORCHESTRA_SYSTEM
    elif dataset in {"gpqa", "gpqa_diamond", "gpqa-diamond"}:
        system = GPQA_ORCHESTRA_SYSTEM
    else:
        system = ORCHESTRA_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            "Continue the SAME MAS optimization trajectory. Use the previous Refiner feedback and execution "
            "result to generate a stronger complete MAS. Preserve the dataset-specific role and no-fake-tool "
            "requirements above. Do not emit a patch; emit one full <mas> XML document.\n\n"
            + json.dumps(payload, ensure_ascii=False)[:30000]
        )},
    ]
