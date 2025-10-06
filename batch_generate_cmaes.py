#!/usr/bin/env python3
"""
batch_generate_cmaes.py
──────────────────────────────────────────────────────────────────────────────
Generate a CMA‑ES optimisation script (`cmaes.py`) in every sub‑folder of
<root> (default: ./trial).

Choose the task with `--task` (one of the keys in TASKS below).  The script
adapts:
  • import path  (taskXX_xxx.step0_scene)
  • base‑environment class
  • derived class name  (<BaseEnv>Tool)
  • correct task string inside __init__
Everything else (trajectory parsing, q_gripper logic, etc.) is identical to
the previous version.
"""

import argparse, ast, json, re, sys, textwrap
import os.path
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
#  Supported tasks  (add more here if you need!)
# ---------------------------------------------------------------------
TASKS = {
    "task01_calabash" : ("CalabashEnv",  "task01_calabash.step0_scene"),
    "task02_reaching" : ("ReachingEnv",  "task02_reaching.step0_scene"),
    "task03_flatten"  : ("FlattenEnv",   "task03_flatten.step0_scene"),
    "task04_holder"   : ("HolderEnv",    "task04_holder.step0_scene"),
    "task05_waterfill": ("WaterfillEnv", "task05_waterfill.step0_scene"),
    "task06_piggy"    : ("PiggyEnv",     "task06_piggy.step0_scene"),
    "task07_lifting"  : ("LiftingEnv",   "task07_lifting.step0_scene"),
    "task08_cutting"  : ("CutEnv",       "task08_cutting.step0_scene"),
    "task09_transport": ("TransportEnv", "task09_transport.step0_scene"),
}

# ╭──────────────────────── AST walker ───────────────────────────────╮
class TrajWalker(ast.NodeVisitor):
    def __init__(self):
        self.vars, self.steps = {}, []

    def _literal(self, node):
        try: return ast.literal_eval(node)
        except Exception: return None

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            val = self._literal(node.value)
            if isinstance(val, tuple) and len(val) == 3:
                self.vars[node.targets[0].id] = val
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = node.func.attr if isinstance(node.func, ast.Attribute) else \
             node.func.id  if isinstance(node.func, ast.Name) else ""
        if fn == "adjust_gripper_pose":
            pos = self._get_pos(node)
            if pos: self.steps.append(("adjust", pos))
        elif fn == "grasp":
            self.steps.append(("grasp", None))
        elif fn == "release":
            self.steps.append(("release", None))
        self.generic_visit(node)

    def _get_pos(self, call):
        # keyword
        for kw in call.keywords:
            if kw.arg == "pos": return self._resolve(kw.value)
        # first positional
        if call.args: return self._resolve(call.args[0])
        return None

    def _resolve(self, node):
        val = self._literal(node)
        if val is not None: return val
        if isinstance(node, ast.Name): return self.vars.get(node.id)
        return None

def parse_trajectory(code: str):
    tree = ast.parse(code)
    w = TrajWalker();  w.visit(tree)
    adjusts = [p for k, p in w.steps if k == "adjust"]
    return w.steps, adjusts
# ╰──────────────────────────────────────────────────────────────────╯

# plan loader (plain python or JSON wrapper)
_JSON_KEYS = ("trajectory_func","trajectory","trajectory_code")
def load_plan(p: Path)->str:
    txt = p.read_text()
    if txt.lstrip()[:1] in "{[":
        try:
            d=json.loads(txt);  # noqa
            if isinstance(d,dict):
                for k in _JSON_KEYS:
                    if k in d and d[k]: return d[k]
        except Exception: pass
    return txt

# placement parser
_PL_RE=re.compile(r"(pos|euler|scale)\s*=\s*(\([^)]+\))")
def parse_place(src:str):
    out={"pos":(0,0,0),"euler":(0,0,0),"scale":(1,1,1)}
    for k,v in _PL_RE.findall(src): out[k]=eval(v)
    return out

# regex fallback (if ast fails)
_ADJ_RE=re.compile(r"adjust_gripper_pose\([^#]*pos\s*=\s*(\([^)]+\))")
_GRASP_RE=re.compile(r"\bgrasp\("); _REL_RE=re.compile(r"\brelease\(")
def regex_parse(lines):
    steps,adj=[],[]
    for ln in lines:
        m=_ADJ_RE.search(ln)
        if m:
            t=eval(m.group(1)); steps.append(("adjust",t)); adj.append(t)
        elif _GRASP_RE.search(ln): steps.append(("grasp",None))
        elif _REL_RE.search(ln):  steps.append(("release",None))
    return steps,adj

# build evaluate_single() body
def build_body(steps):
    out,idx=[],0
    for k,v in steps:
        if k=="adjust":
            out.append(
              f"adjust_gripper_pose_without_plan(self.scene, self.xarm,"
              f" pos=poss[{idx}], quat=init_quat, save_img=self.save_img,"
              f" q_gripper=q_gripper)")
            idx+=1
        elif k=="grasp":
            out+=["my_grasp(self.scene, self.xarm, self.tool, self.save_img)",
                  "q_gripper = 0.044"]
        elif k=="release":
            out+=["release(self.scene, self.xarm, self.save_img)",
                  "q_gripper = 0"]
    return "\n".join(out)

# template (place‑holders are {{}})
TEMPLATE = """
import numpy as np
import genesis as gs
import os

from {module} import {base_env}
from utils.api_manipulate import my_grasp, release, adjust_gripper_pose_without_plan, euler_to_quat


class {derived}({base_env}):
    def __init__(self, task='{task}', log_dir=os.path.join('{task}_tools', 'cmaes{design_id}')):
        super().__init__(task, log_dir=log_dir)

    # -----------------------------------------------------------------
    # tool placement
    # -----------------------------------------------------------------
    def add_tools_for_task(self):
        self.tool = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file='{design_id}.obj',
                scale={scale},
                pos=(0.3, -0.3, {pos2}+0.01),
                euler={euler},
                fixed=False,
                collision=True,
            ),
            surface=gs.surfaces.Default(color=(0.1,0.1,0.1)),
        )

    # -----------------------------------------------------------------
    # CMA-ES evaluation
    # -----------------------------------------------------------------
    def evaluate(self, poss):
        poss = poss.reshape(self.n_envs, {n_steps}, 3)
        rewards = np.zeros(self.n_envs)
        for i in range(self.n_envs):
            rewards[i] = self.evaluate_single(poss[i])
        return rewards

    def evaluate_single(self, poss):
        self.scene.reset()
        q_gripper = 0
        init_quat = euler_to_quat({euler})

        # auto‑generated trajectory
{body}
        reward = self.metric()
        return reward


# ------------------------------- driver ------------------------------
env = {derived}()

dim, sigma, n_envs, iters = {dim}, {sigma}, {n_envs}, {iters}
init_params = np.array({init_params}).flatten()

env.set_cmaes_params(dim, init_params, range={rng}, sigma=sigma,
                     n_envs=n_envs, iters=iters)
env.build_scene_for_evaluation()
# env.n_envs = n_envs
# env.optimize()
env.reset()
env.save_img()
""".lstrip()

def generate(design_json:Path, plan_code:str, out:Path,
             task:str, n_envs=15,iters=50,rng=0.1,sigma=0.02):
    base_env, module = TASKS[task]
    derived = f"{base_env}Tool"

    # import pdb; pdb.set_trace()

    d = json.loads(design_json.read_text())
    place = parse_place(d["placement_func"])
    euler,pos,scale = place["euler"],place["pos"],place["scale"]
    # did = re.search(r'design(\d+)\.json',design_json.name)
    # design_id = did.group(1) if did else "1"

    # print('design_json:', str(design_json))
    # print('design_json.split()', str(design_json).split('/'))

    design_id = int(str(design_json).split("/")[-2])
    # print('design_id:', design_id)

    try:
        steps,adjusts = parse_trajectory(plan_code)
    except SyntaxError:
        steps,adjusts = regex_parse(plan_code.splitlines())

    body = build_body(steps)
    n_steps=len(adjusts); dim=max(1,3*n_steps)

    script=TEMPLATE.format(
        module=module, base_env=base_env, derived=derived, task=task,
        design_id=design_id, scale=scale, pos2=pos[2], euler=euler,
        n_steps=max(1,n_steps), body=textwrap.indent(body," "*8),
        dim=dim, sigma=sigma, n_envs=n_envs, iters=iters,
        init_params=adjusts or [(0,0,0)], rng=rng,
        date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    out.write_text(script)
    print(f"[✓] wrote {out.relative_to(out.parent.parent)}")

# batch over trials
def process(trial:Path,task:str):
    plan=trial/"plan.txt"; designs=sorted(trial.glob("design*.json"))
    if not plan.exists() or not designs:
        print(f"[SKIP] {trial.name}: missing plan.txt or design*.json"); return
    try:
        code=load_plan(plan)
        generate(designs[-1],code,trial/"cmaes.py",task)
    except Exception as e:
        print(f"[ERROR] {trial.name}: {e}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", help="root dir (default: ./trial)")
    ap.add_argument("--task", choices=TASKS.keys(),
                    help="which task to generate for")
    args=ap.parse_args()
    args.root = os.path.join(args.task, args.root)
    root=Path(args.root).resolve()
    if not root.is_dir(): sys.exit(f"{root} is not a directory")

    for td in sorted(p for p in root.iterdir() if p.is_dir()): process(td,args.task)

if __name__=="__main__": main()
