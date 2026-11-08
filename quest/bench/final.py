"""Final validation: the shipped continuity.WristFlipGuard inside the bench IK, with the shipped defaults."""
import sys,pathlib; sys.path.insert(0,str(pathlib.Path(__file__).parent))
sys.path.insert(0,"C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/teleop/quest")
import numpy as np, bench_ik as B, fk as F, resp as R, continuity as C
class ShippedIK(B.IK):
    def __init__(s,**kw): super().__init__(**kw); s.g=C.WristFlipGuard(); s.rebases=0
    def engage(s,q): super().engage(q); s.g.reset()
    def step(s,target,q,dt):
        q=np.asarray(q,float); sol=super().step(s.g.apply(target),q,dt); T0,T1=F.fk(q),F.fk(sol)
        if s.g.observe(q,sol,T0,T1,dt): sol=super().step(s.g.rebase(T0,target),q,dt)
        s.rebases=s.g.rebases; return sol
CAPS=[2.5,2.5,2.5,4,4,4]
# 1) recorded flip events
ev=B.events()
for name,mk,hz in (("kit 30 Hz",lambda:B.IK(),30),("quest_collect before",lambda:B.IK(cap=2.5,damping=1e-2),60),("shipped",lambda:ShippedIK(cap=CAPS,damping=1e-2),60)):
    rows=[];reb=0
    for _,q0,q1 in ev:
        ik=mk(); r=B.run(ik,q0,F.fk(q1),hz); rows.append(r); reb+=getattr(ik,"rebases",0)
    dev=np.array([max(r["maxdev"][3],r["maxdev"][5]) for r in rows]); P=np.array([r["p"] for r in rows])
    print(f"[events] {name:22s} worst wrist excursion med {np.median(dev):4.0f} p90 {np.percentile(dev,90):4.0f} max {dev.max():4.0f} deg | >90 deg: {(dev>90).sum()}/{len(dev)} | pos err max {P.max():.2f} cm | rebases {reb}")
# 2) responsiveness on human motion
B_IK=B.IK
R.CFGS["shipped"]=(60,dict(ik=dict(cap=CAPS,damping=1e-2),filter=(0.7,10.0,3.0),noise_mm=1.5,noise_deg=0.4,shipped=True))
orig=R.pipeline
def pipeline(cfg,Ts,q0,hz,rng):
    if cfg.get("shipped"):
        R.B.IK=ShippedIK
        class PF(R.PoseFilter.__class__ if False else object): pass
        import sweep_pf
        R.PoseFilter=lambda a,b,c: sweep_pf.PF2(a,b,0.5,c)
        try: return orig(cfg,Ts,q0,hz,rng)
        finally: R.B.IK=B_IK; R.PoseFilter=sweep_pf.ORIG
    return orig(cfg,Ts,q0,hz,rng)
R.pipeline=pipeline
trajs=R.truth_trajs()
for name in ("kit (30 Hz, cap 1.5, no filter)","quest_collect today","shipped"):
    hz,cfg=R.CFGS[name]; rng=np.random.default_rng(0); acc=[];reb=0
    for Q in trajs:
        Ts,q0=R.upsample(Q,hz//30); out,r=R.pipeline(cfg,Ts,q0,hz,rng); acc.append(R.metrics(Ts,out,hz)); reb+=r
    m={k:np.mean([a[k] for a in acc]) for k in acc[0]}
    print(f"[human] {name:32s} pos err med {m['perr']:.1f} p95 {m['p95']:.1f} mm | rot err med {m['rerr']:.1f} p95 {m['r95']:.1f} deg | lag pos {m['plag']:.0f} rot {m['rlag']:.0f} ms | jitter {m['jit']:.2f} mm rot {m['rjit']:.3f} | rebases {reb}")
