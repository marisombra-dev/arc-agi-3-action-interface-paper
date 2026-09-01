from __future__ import annotations
import ast, io, os, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research'/'sonpham-wm'/'ARC3-Inference'
source=ast.parse((ROOT/'model_harness'/'validate_resource_aware_frontier_policy_e2e.py').read_text(encoding='utf-8'))
OVERLAYS=[]
for node in source.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OVERLAYS' for t in node.targets):
        OVERLAYS=list(ast.literal_eval(node.value)); break
assert OVERLAYS
if 'apply_interaction_frontier_overlay.py' not in OVERLAYS: OVERLAYS.append('apply_interaction_frontier_overlay.py')
blob=subprocess.check_output(['git','-C',str(BASE),'archive','--format=zip','33dcb90'])
with tempfile.TemporaryDirectory() as td:
    target=Path(td)/'src'; target.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf: zf.extractall(target)
    for name in OVERLAYS:
        subprocess.check_call([sys.executable,str(ROOT/'model_harness'/name),str(target)],cwd=ROOT,stdout=subprocess.DEVNULL)
    subprocess.check_call([sys.executable,'-m','py_compile',str(target/'inference/agent/interaction_frontier.py'),str(target/'inference/agent/tool_agent.py')])
    sys.path.insert(0,str(target))
    os.environ['LOCAL_ANALYZER_MODEL_ID']='fake/model'
    os.environ['LOCAL_ANALYZER_BASE_URL']='http://127.0.0.1:1/v1'
    from inference.agent import runtime_state as rs, tool_agent as ta

    grid=[[0 for _ in range(16)] for _ in range(16)]
    for row,col in ((3,3),(3,10),(10,3)):
        for dr in (0,1):
            for dc in (0,1): grid[row+dr][col+dc]=2
    frame=rs.Frame(grid=tuple(tuple(row) for row in grid),step=0,level=1)
    state=Path(td)/rs.RUNTIME_STATE_FILENAME
    rs.write_runtime_state(state,current_frame=frame,history=[])
    agent=ta.ToolAgent(model='fake/model',base_url='http://127.0.0.1:1/v1',provider='vllm')
    agent._current_valid_actions=['MOUSE']
    agent._ensure_session(state)
    prompt=agent._build_user_prompt(0,valid_actions=['MOUSE'],current_frame=frame,history_entries=[],previous_step_summary=None)
    assert 'HOST_INTERACTION: ranked source-free MOUSE candidates:' in prompt
    assert agent._patricia_current_interventions
    key,item=next(iter(agent._patricia_current_interventions.items()))
    _,row,col=key
    calls=[]
    def step_env(payload):
        calls.append(payload['actions'][0])
        changed=[list(values) for values in grid]
        changed[row][col]=3 if changed[row][col] != 3 else 4
        after=rs.Frame(grid=tuple(tuple(values) for values in changed),step=1,level=1)
        rs.write_runtime_state(state,current_frame=after,history=[rs.HistoryEntry(action='MOUSE',frame=after)])
        return {'executed':True,'action_num':1,'level':1,'reward':0.0,'state':'NOT_FINISHED','valid_actions':['MOUSE'],'board_changed':True,'done':False,'level_completed':False,'game_over':False,'run_complete':False,'action_display':'MOUSE','executed_actions':['MOUSE'],'executed_count':1,'transition_signature':{'cells':1,'bbox':[row,col,row,col],'pairs':['2>3:1'],'motions':[]}}
    agent._step_env_callback=step_env
    result=agent._run_action_tool(state,{'action':'MOUSE','row':row,'col':col,'world_model':'{}'})
    assert result.step_executed
    assert calls==[{'action':'MOUSE','row':row,'col':col}]
    assert len(agent._patricia_interaction_observations)==1
    obs=agent._patricia_interaction_observations[0]
    assert obs['changed_cells']==1 and obs['scope']=='target_local'
    assert agent._patricia_pending_intervention is not None
    assert agent._last_action_result['interaction_observation']['changed_cells']==1
    after_frame,_=rs.load_runtime_state(state)
    prompt2=agent._build_user_prompt(1,valid_actions=['MOUSE'],current_frame=after_frame,history_entries=[],previous_step_summary=agent._last_action_result)
    assert 'HOST_INTERACTION_RESULT:' in prompt2
    assert 'destination/follow-up target' in prompt2
    # A tiny remote/HUD-only change on an object-like click may still represent latent selection.
    rs.write_runtime_state(state,current_frame=frame,history=[])
    latent=ta.ToolAgent(model='fake/model',base_url='http://127.0.0.1:1/v1',provider='vllm')
    latent._current_valid_actions=['MOUSE']
    latent._ensure_session(state)
    latent._build_user_prompt(0,valid_actions=['MOUSE'],current_frame=frame,history_entries=[],previous_step_summary=None)
    latent_item=latent._patricia_current_interventions[("MOUSE",row,col)]
    def latent_step(payload):
        changed=[list(values) for values in grid]; changed[0][0]=7
        after=rs.Frame(grid=tuple(tuple(values) for values in changed),step=1,level=1)
        rs.write_runtime_state(state,current_frame=after,history=[rs.HistoryEntry(action='MOUSE',frame=after)])
        return {'executed':True,'action_num':1,'level':1,'reward':0.0,'state':'NOT_FINISHED','valid_actions':['MOUSE'],'board_changed':True,'done':False,'level_completed':False,'game_over':False,'run_complete':False,'action_display':'MOUSE','executed_actions':['MOUSE'],'executed_count':1,'transition_signature':{'cells':1,'bbox':[0,0,0,0],'pairs':['0>7:1'],'motions':[]}}
    latent._step_env_callback=latent_step
    latent_result=latent._run_action_tool(state,{'action':'MOUSE','row':row,'col':col,'world_model':'{}'})
    assert latent_result.step_executed
    latent_obs=latent._patricia_interaction_observations[0]
    assert latent_obs['causality']=='uncertain_ambient' and latent_obs['followup_reason']=='latent_selection'
    assert latent._patricia_pending_intervention['_followup_reason']=='latent_selection'
    latent_frame,_=rs.load_runtime_state(state)
    latent_prompt=latent._build_user_prompt(1,valid_actions=['MOUSE'],current_frame=latent_frame,history_entries=[],previous_step_summary=latent._last_action_result)
    assert 'may have armed a latent selection' in latent_prompt
    assert 'causality=uncertain_ambient' in latent_prompt
    print('INTERACTION_FRONTIER_E2E_PASS',row,col,obs['transition_class'])
