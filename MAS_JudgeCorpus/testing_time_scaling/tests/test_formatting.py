from testing_time_scaling.formatting import make_xmls, rm_text


def test_rm_prefixes_and_levels():
    mas={"topology":"dag","agents":[{"id":"a","role":"research","execution_mode":"multi_turn_search","tools":["bm25"]}],"edges":[]}
    traj={"agent_traces":[{"agent_id":"a","output":"found evidence","search_mode":"bm25","search_rounds":[]}],"final_output":"Paris","success":True}
    orch,traj_xml,orch_text,traj_text=make_xmls("Where?",mas,traj)
    assert orch_text.startswith("Task:\nWhere?\n\nCandidate MAS:\n<mas>")
    assert traj_text.startswith("Task:\nWhere?\n\nCandidate MAS with per-agent execution:\n<mas>")
    assert "<execution>" not in orch and "<execution>" in traj_xml
