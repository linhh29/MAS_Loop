import pytest

from mas_loop_rl.prompts import editor_prompt, orchestra_prompt
from mas_loop_rl.schema import apply_patch, normalize_editor_action, normalize_mas


SOURCE = {
    "agents": [
        {"id": "agent_0", "role": "Find evidence", "tools": [], "budget": 1, "execution_mode": "llm_only"},
        {"id": "agent_1", "role": "Answer", "tools": [], "budget": 1, "execution_mode": "llm_only"},
    ],
    "edges": [{"from": "agent_0", "to": "agent_1", "type": "data_flow"}],
    "aggregation": "agent_1", "topology": "dag", "max_cost": 2, "max_iterations": 3,
}


def test_xml_mas_and_escaping_parse():
    text = '''prefix <mas topology="dag" aggregation="agent_1" max_cost="2" max_iterations="3"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Find A &amp; B</role><tools><tool>bm25</tool></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Answer</role><tools /></agent></agents><edges><edge from="agent_0" to="agent_1" type="data_flow" /></edges></mas>'''
    mas = normalize_mas(text)
    assert mas["agents"][0]["role"] == "Find A & B"
    assert mas["agents"][0]["tools"] == ["bm25"]


def test_xml_mas_repairs_redundant_empty_container_closes():
    text = '''<mas topology="dag" aggregation="agent_1" max_cost="2" max_iterations="3"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Find evidence</role><tools><tool>bm25</tool></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Answer</role><tools /></tools></agent></agents><edges><edge from="agent_0" to="agent_1" type="data_flow" /></edges></mas>'''
    mas = normalize_mas(text)
    assert mas["aggregation"] == "agent_1"
    assert mas["agents"][1]["tools"] == []


def test_xml_mas_repairs_one_missing_final_agent_close():
    text = '''<mas topology="dag" aggregation="agent_1" max_cost="2" max_iterations="3"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Find evidence</role><tools><tool>bm25</tool></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Answer</role><tools /></tools></agents><edges><edge from="agent_0" to="agent_1" type="data_flow" /></edges></mas>'''
    mas = normalize_mas(text)
    assert [agent["id"] for agent in mas["agents"]] == ["agent_0", "agent_1"]


def test_xml_mas_repairs_missing_agents_container_close_before_edges():
    text = '''<mas topology="dag" aggregation="agent_1" max_cost="2" max_iterations="3"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Find evidence</role><tools><tool>bm25</tool></tools></agent><agent id="agent_1" budget="1" execution_mode="llm_only"><role>Answer</role><tools></tools></agent><edges><edge from="agent_0" to="agent_1" type="data_flow" /></edges></mas>'''
    mas = normalize_mas(text)
    assert [agent["id"] for agent in mas["agents"]] == ["agent_0", "agent_1"]
    assert mas["edges"] == [{"from": "agent_0", "to": "agent_1", "type": "data_flow"}]


def test_xml_mas_repairs_bare_ampersand_without_changing_text():
    text = '''<mas topology="dag" aggregation="agent_0" max_cost="1" max_iterations="1"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Find A & B</role><tools><tool>bm25</tool></tools></agent></agents><edges></edges></mas>'''
    mas = normalize_mas(text)
    assert mas["agents"][0]["role"] == "Find A & B"


def test_xml_mas_repairs_unescaped_numeric_comparison_without_changing_text():
    text = '''<mas topology="dag" aggregation="agent_0" max_cost="1" max_iterations="1"><agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search"><role>Find player with <250 centuries.</role><tools><tool>bm25</tool></tools></agent></agents><edges></edges></mas>'''
    mas = normalize_mas(text)
    assert mas["agents"][0]["role"] == "Find player with <250 centuries."


def test_xml_refiner_repairs_spaced_numeric_comparison_without_changing_text():
    text = '''<refiner><stage1_localization source_status="actionable_failure" target_type="agent" target_id="agent_0" failure_type="incorrect_ordering"><evidence><item agent_id="agent_0" execution_index="0" field="output_summary" location="node_cards[0].output_summary">Order: 4 (NO2) < 6 (COOH) < 2 (COOC2H5).</item></evidence></stage1_localization><stage2_causal_model><causal_chain><step>The ordering was wrong.</step></causal_chain><causal_explanation>The source answer used the wrong ordering.</causal_explanation><counterfactual>A corrected ordering would change the answer.</counterfactual><downstream_impact>The final option was wrong.</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="0.8"><repair_goal>Correct the ordering.</repair_goal><expected_effect>Produce the correct option.</expected_effect><risk>Low.</risk></stage3_repair_plan><patch><operation op="rewrite_role" agent_id="agent_0"><role>Check substituent ordering carefully.</role></operation></patch></refiner>'''
    patch = normalize_editor_action(text, src_mas=SOURCE)
    assert patch["stage1_localization"]["evidence"][0]["observation"] == (
        "Order: 4 (NO2) < 6 (COOH) < 2 (COOC2H5)."
    )
    assert patch["operations"] == [
        {"op": "rewrite_role", "agent_id": "agent_0", "role": "Check substituent ordering carefully."}
    ]


def test_xml_refiner_repairs_spaced_identifier_comparison_without_changing_text():
    text = '''<refiner><stage1_localization source_status="actionable_failure" target_type="agent" target_id="agent_0" failure_type="incorrect_ordering"><evidence><item agent_id="agent_0" execution_index="0" field="output_summary" location="node_cards[0].output_summary">The observed relation is λ2 < λ1 while x < y remains possible.</item></evidence></stage1_localization><stage2_causal_model><causal_chain><step>The ordering was wrong.</step></causal_chain><causal_explanation>The source answer used the wrong variable ordering.</causal_explanation><counterfactual>A corrected ordering would change the answer.</counterfactual><downstream_impact>The final option was wrong.</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="0.8"><repair_goal>Correct the ordering.</repair_goal><expected_effect>Produce the correct option.</expected_effect><risk>Low.</risk></stage3_repair_plan><patch><operation op="rewrite_role" agent_id="agent_0"><role>Check λ2 < λ1 carefully.</role></operation></patch></refiner>'''
    patch = normalize_editor_action(text, src_mas=SOURCE)
    assert patch["stage1_localization"]["evidence"][0]["observation"] == (
        "The observed relation is λ2 < λ1 while x < y remains possible."
    )
    assert patch["operations"] == [
        {"op": "rewrite_role", "agent_id": "agent_0", "role": "Check λ2 < λ1 carefully."}
    ]


def test_xml_refiner_repairs_real_missing_downstream_impact_close_with_add_verifier():
    # Regression from GPQA step 21, sample gpqa_diamond_164. The model emitted
    # complete downstream-impact text but omitted only its leaf closing tag.
    text = '''<refiner><stage1_localization source_status="actionable_failure" target_type="agent" target_id="agent_2" failure_type="incorrect_final_answer"><evidence><item agent_id="agent_2" execution_index="2" field="output_summary" location="node_cards[2].output_summary" round="0">\\boxed{D}</item></evidence></stage1_localization><stage2_causal_model><causal_chain><step>agent_0 generated evidence suggesting D as the answer</step><step>agent_1 reconciled evidence and confirmed D</step><step>agent_2 directly output \\boxed{D} without further validation</step></causal_chain><causal_explanation>The final answer was directly output by agent_2 without additional verification, leading to an incorrect final output despite prior consensus on D.</causal_explanation><counterfactual>If agent_2 had performed a final validation step, the answer might have been correctly confirmed.</counterfactual><downstream_impact>The incorrect final answer \\boxed{D} was produced, but the score was 0.7, indicating partial correctness.</stage2_causal_model><stage3_repair_plan confidence="high"><repair_goal>Ensure final answer is validated before output</repair_goal><expected_effect>Reduces risk of incorrect final output by adding a validation step</expected_effect><risk>Minimal, as the current answer is likely correct but needs confirmation</risk></stage3_repair_plan><patch><operation op="add_verifier" agent_id="agent_3" budget="1"><role>Final Answer Validator</role></operation><operation op="add_edge" from="agent_2" to="agent_3" type="data_flow" /></patch></refiner>'''
    source = {
        "agents": [
            *SOURCE["agents"],
            {"id": "agent_2", "role": "Return the answer", "tools": [], "budget": 1, "execution_mode": "llm_only"},
        ],
        "edges": [
            {"from": "agent_0", "to": "agent_1", "type": "data_flow"},
            {"from": "agent_1", "to": "agent_2", "type": "data_flow"},
        ],
        "aggregation": "agent_2", "topology": "dag", "max_cost": 3, "max_iterations": 3,
    }
    patch = normalize_editor_action(text, src_mas=source)
    assert patch["stage2_causal_model"]["downstream_impact"].endswith("partial correctness.")
    assert patch["operations"] == [
        {"op": "add_verifier", "agent_id": "agent_3", "role": "Final Answer Validator", "budget": 1},
        {"op": "add_edge", "from": "agent_2", "to": "agent_3", "type": "data_flow"},
    ]
    updated, errors = apply_patch(source, patch)
    assert errors == []
    assert updated["aggregation"] == "agent_3"
    assert updated["agents"][-1]["id"] == "agent_3"


def test_xml_refiner_aliases_translate_and_apply():
    text = '''<refiner><stage1_localization source_status="actionable_failure" target_type="agent" target_id="AGENT-0" failure_type="capability"><evidence><item location="agent_0.output">No retrieval</item></evidence></stage1_localization><stage2_causal_model><causal_chain><step>Missing evidence</step></causal_chain><causal_explanation>No tool</causal_explanation><counterfactual>Retrieval adds evidence</counterfactual><downstream_impact>Better answer</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="0.8"><repair_goal>Add retrieval</repair_goal><expected_effect>Evidence</expected_effect><risk>Cost</risk></stage3_repair_plan><patch><operation op="change_role" agent_id="AGENT-0"><role>Retrieve &amp; verify evidence</role></operation><operation op="add_tool" agent="agent_0" new_tool="bm25" /></patch></refiner>'''
    patch = normalize_editor_action(text, src_mas=SOURCE)
    assert [op["op"] for op in patch["operations"]] == ["rewrite_role", "replace_tool"]
    updated, errors = apply_patch(SOURCE, patch)
    assert errors == []
    assert updated["agents"][0]["tools"] == ["bm25"]
    assert patch["translation"]["status"] == "applied"


def test_json_backward_compatibility():
    patch = normalize_editor_action({"patch": [{"op": "accept"}]}, src_mas=SOURCE)
    assert patch["operations"] == [{"op": "accept"}]


def test_partial_patch_keeps_valid_operation():
    patch = normalize_editor_action({"patch": [{"op": "rewrite_role", "agent_id": "agent_0", "role": "Better"}, {"op": "unknown"}]}, src_mas=SOURCE)
    assert patch["operations"] == [{"op": "rewrite_role", "agent_id": "agent_0", "role": "Better"}]
    assert patch["translation"]["dropped_operations"]


def test_dtd_is_rejected_without_json_fallback():
    with pytest.raises(ValueError):
        normalize_mas('<!DOCTYPE mas [<!ENTITY x "bad">]><mas><agents /></mas>')


def test_prompts_are_xml_first():
    orchestra = orchestra_prompt("x")[1]["content"]
    assert "<mas> XML" in orchestra
    assert "<tools></tools>" in orchestra
    assert "Always close every <agent>" in orchestra
    prompt = editor_prompt("x", SOURCE, {"final_output": "x", "agent_outputs": []})
    assert "<refiner>" in prompt[1]["content"]
    assert "OUTPUT must be XML" in prompt[1]["content"]


def test_xml_refiner_parses_grounding_attributes_and_rejects_entity_payload():
    text = '''<refiner><stage1_localization source_status="actionable_failure" target_type="agent" target_id="agent_0" failure_type="missing_evidence"><evidence><item agent_id="agent_0" execution_index="0" round="1" field="query" location="node_cards[0].optional_capability_signals.retrieval.round_summaries[0].query">The first query is too generic.</item></evidence></stage1_localization><stage2_causal_model><causal_chain><step>Weak query</step></causal_chain><causal_explanation>The query misses the key entity.</causal_explanation><counterfactual>A specific query retrieves evidence.</counterfactual><downstream_impact>The answer node receives evidence.</downstream_impact></stage2_causal_model><stage3_repair_plan confidence="0.8"><repair_goal>Make retrieval task-specific.</repair_goal><expected_effect>Relevant evidence.</expected_effect><risk>Low.</risk></stage3_repair_plan><patch><operation op="rewrite_role" agent_id="agent_0"><role>Retrieve task-specific evidence.</role></operation></patch></refiner>'''
    patch = normalize_editor_action(text, src_mas=SOURCE)
    evidence = patch["stage1_localization"]["evidence"][0]
    assert evidence == {
        "location": "node_cards[0].optional_capability_signals.retrieval.round_summaries[0].query",
        "observation": "The first query is too generic.",
        "agent_id": "agent_0",
        "field": "query",
        "execution_index": 0,
        "round": 1,
    }
    with pytest.raises(ValueError):
        normalize_editor_action('<!DOCTYPE refiner [<!ENTITY x SYSTEM "file:///etc/passwd">]><refiner>&x;</refiner>')


def test_editor_prompt_evidence_shape_has_nonempty_observation_body():
    prompt = editor_prompt("x", SOURCE, {"final_output": "x", "agent_outputs": []})[1]["content"]
    evidence_start = prompt.index('<evidence><item agent_id="..."')
    evidence_end = prompt.index("</item></evidence>", evidence_start)
    body = prompt[prompt.index(">", evidence_start) + 1:evidence_end].strip()
    assert body
    assert "Concrete task-specific observation" in body
    assert "never emit an empty <evidence><item></item> body" in prompt
