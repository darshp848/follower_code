"""Responsiveness and jitter bench: human wrist/hand motion (leader recordings) + Quest-like noise -> filter -> clutch -> IK."""
import sys,pathlib,ast,math; sys.path.insert(0,str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, mink
import bench_ik as B, fk as F, guard as G
QC=pathlib.Path("C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/teleop/quest/quest_collect.py")
src=QC.read_text(); tree=ast.parse(src); ns={"np":np,"math":math,"npt":np.typing}
import numpy.typing
for node in tree.body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ("axis_angle","rotation_about","OneEuro","PoseFilter"):
        exec(compile(ast.Module([node],[]),"qc","exec"),ns)
OneEuro,PoseFilter,axis_angle,rotation_about=ns["OneEuro"],ns["PoseFilter"],ns["axis_angle"],ns["rotation_about"]
def ang(R): return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))

def truth_trajs():
    out=[]
    for f in sorted((B.H/"leader-split-v1-frozen").glob("data/chunk-000/*.parquet")):
        s=np.stack(pd.read_parquet(f)["observation.state"]).astype(float)[:,7:13]
        out.append(s)
    return out
def upsample(Q,k):
    T=[F.fk(q) for q in Q]; out=[]
    for i in range(len(T)-1):
        for j in range(k): out.append(B.se3_interp(T[i],T[i+1],j/k))
    return out, Q[0]

def pipeline(cfg, Ts, q0, hz, rng):
    dt=1/hz; filt=PoseFilter(*cfg["filter"]) if cfg.get("filter") else None
    ik=G.GuardIK(**cfg["ik"]) if cfg.get("guard") else B.IK(**cfg["ik"])
    if cfg.get("guard"): ik.window_s,ik.min_travel,ik.ratio=cfg["guard"]
    ik.engage(q0); q=q0.copy(); tool_start=F.fk(q0); hand_start=None; out=[]
    db=math.radians(cfg.get("deadband_deg",0.0))
    for Tt in Ts:
        H=Tt.copy(); H[:3,3]+=rng.normal(0,cfg["noise_mm"]/1000,3)
        H[:3,:3]=rotation_about(*(lambda v:(v/np.linalg.norm(v),np.linalg.norm(v)))(rng.normal(0,math.radians(cfg["noise_deg"]),3)+1e-12))@H[:3,:3]
        if filt is not None: H=filt(H,dt)
        if hand_start is None: hand_start=H.copy()
        tgt=np.eye(4); tgt[:3,3]=tool_start[:3,3]+(H[:3,3]-hand_start[:3,3])
        ax,an=axis_angle(H[:3,:3]@hand_start[:3,:3].T); an=max(0.0,an-db)
        tgt[:3,:3]=rotation_about(ax,an)@tool_start[:3,:3]
        q=ik.step(tgt,q,dt); out.append(F.fk(q))
    return out, getattr(ik,"rebases",0)

def metrics(Ts,out,hz):
    P=np.array([T[:3,3] for T in Ts]); Po=np.array([T[:3,3] for T in out])
    perr=np.linalg.norm(P-Po,axis=1)*1000; rerr=np.array([ang(a[:3,:3]@b[:3,:3].T) for a,b in zip(Ts,out)])
    # rotation lag: shift output vs truth rotation to minimise rot error
    best=min(range(0,int(0.3*hz)),key=lambda s: np.mean([ang(Ts[i][:3,:3]@out[i+s][:3,:3].T) for i in range(0,len(Ts)-s,3)]))
    pbest=min(range(0,int(0.3*hz)),key=lambda s: np.mean(np.linalg.norm(P[:len(P)-s]-Po[s:],axis=1)))
    # jitter: high-frequency content = residual from a 5-sample moving average of the output
    k=max(3,int(hz/12)); ker=np.ones(k)/k
    sm=np.stack([np.convolve(Po[:,i],ker,"same") for i in range(3)],1)
    jit=np.linalg.norm((Po-sm)[k:-k],axis=1).mean()*1000
    rv=np.array([ang(out[i+1][:3,:3]@out[i][:3,:3].T) for i in range(len(out)-1)])
    rjit=np.mean(np.abs(np.diff(rv)))  # deg per tick change of turn rate
    return dict(perr=np.median(perr),p95=np.percentile(perr,95),rerr=np.median(rerr),r95=np.percentile(rerr,95),rlag=best/hz*1000,plag=pbest/hz*1000,jit=jit,rjit=rjit*hz/60)

CFGS={
 "kit (30 Hz, cap 1.5, no filter)": (30, dict(ik=dict(), noise_mm=1.5, noise_deg=0.4)),
 "quest_collect today": (60, dict(ik=dict(cap=2.5,damping=1e-2), filter=(0.7,10.0,0.5), deadband_deg=2.0, noise_mm=1.5, noise_deg=0.4)),
 "proposed": (60, dict(ik=dict(cap=[2,2,2,4,4,4],damping=1e-2), filter=(1.0,10.0,4.0), guard=(0.3,15,0.3), noise_mm=1.5, noise_deg=0.4)),
}
if __name__=="__main__":
    sel=sys.argv[1:] or list(CFGS)
    trajs=truth_trajs()
    for name in sel:
        hz,cfg=CFGS[name]; rng=np.random.default_rng(0); acc=[];reb=0
        for Q in trajs:
            Ts,q0=upsample(Q,hz//30); out,r=pipeline(cfg,Ts,q0,hz,rng); acc.append(metrics(Ts,out,hz)); reb+=r
        m={k:np.mean([a[k] for a in acc]) for k in acc[0]}
        print(f"{name:34s} pos err med {m['perr']:.1f} p95 {m['p95']:.1f} mm | rot err med {m['rerr']:.1f} p95 {m['r95']:.1f} deg | lag pos {m['plag']:.0f} rot {m['rlag']:.0f} ms | jitter {m['jit']:.2f} mm, rot {m['rjit']:.3f} | rebases {reb}")
