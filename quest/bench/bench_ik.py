"""Replays the recorded wrist-flip events through the kit's IK and through variants."""
import sys, numpy as np, pandas as pd, pathlib, mink, mujoco
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import fk as F
H=pathlib.Path("C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/nuc_data/hackathon_data")
NAMES=[f"joint{i}" for i in range(1,7)]
M=F.M
jid=[M.joint(n).id for n in NAMES]; qadr=np.array([M.jnt_qposadr[i] for i in jid])
low=np.array([M.jnt_range[i][0] for i in jid])+0.05; high=np.array([M.jnt_range[i][1] for i in jid])-0.05

class IK:
    def __init__(s, cap=1.5, damping=1e-3, ocost=0.5, posture=None, anchor="engage", leak_s=2.0):
        s.cap=np.broadcast_to(np.asarray(cap,float),(6,)).copy(); s.damping=damping
        s.conf=mink.Configuration(M); s.task=mink.FrameTask("tcp","site",position_cost=1.0,orientation_cost=ocost,lm_damping=damping)
        s.limits=[mink.ConfigurationLimit(M,min_distance_from_limits=0.0), mink.VelocityLimit(M,{n:float(c) for n,c in zip(NAMES,s.cap)})]
        s.posture=None
        if posture is not None:
            s.posture=mink.PostureTask(M,cost=np.asarray(posture,float)); s.anchor=anchor; s.leak_s=leak_s; s.ref=None
    def engage(s,q): s.ref=None if s.posture is None else np.asarray(q,float).copy()
    def step(s,target,q,dt):
        seed=np.clip(np.asarray(q,float),low,high); qq=s.conf.q.copy(); qq[qadr]=seed; s.conf.update(qq)
        s.task.set_target(mink.SE3.from_matrix(target)); tasks=[s.task]
        if s.posture is not None:
            if s.ref is None: s.ref=seed.copy()
            if s.anchor=="leaky": s.ref += (seed-s.ref)*min(1.0,dt/s.leak_s)
            full=s.conf.q.copy(); full[qadr]=s.ref; s.posture.set_target(full); tasks.append(s.posture)
        di=dt/3
        try:
            for _ in range(3):
                v=mink.solve_ik(s.conf,tasks,di,"daqp",damping=s.damping,limits=s.limits); s.conf.integrate_inplace(v,di)
        except mink.NoSolutionFound: return np.asarray(q,float).copy()
        sol=np.clip(s.conf.q[qadr],low,high); m=s.cap*dt
        return np.clip(sol,seed-m,seed+m)

def se3_interp(T0,T1,a):
    P0,P1=mink.SE3.from_matrix(T0),mink.SE3.from_matrix(T1)
    return (P0 @ mink.SE3.exp((P0.inverse()@P1).log()*a)).as_matrix()
def err(T,Tt):
    p=np.linalg.norm(T[:3,3]-Tt[:3,3]); R=T[:3,:3].T@Tt[:3,:3]; r=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))); return p,r

def events():
    out=[]
    for ds in ("unplug-v1","unplug-split-v1"):
        for f in sorted((H/ds).glob("data/chunk-000/*.parquet")):
            a=np.stack(pd.read_parquet(f)["action"]).astype(float)
            for off in (7,0):
                A=a[:,off:off+6]
                if np.ptp(A,axis=0).max()<0.05: continue
                sat=(np.abs(np.diff(A,axis=0))>=0.049).any(1); i=0
                while i<len(sat):
                    if not sat[i]: i+=1; continue
                    j=i
                    while j<len(sat) and sat[j]: j+=1
                    if j-i>=8: out.append((f"{ds}/{f.stem[-2:]}{'RL'[off==0]}@{i/30:.1f}",A[i],A[min(j+3,len(A)-1)]))
                    i=j
    return out

def run(ik,q0,Tgoal,hz,ramp_s=0.3,hold_s=2.5):
    dt=1/hz; q=q0.copy(); T0=F.fk(q0); ik.engage(q0); wr=0; trav=np.zeros(6); conv=None
    for k in range(int((ramp_s+hold_s)*hz)):
        Tt=se3_interp(T0,Tgoal,min(1.0,k*dt/ramp_s)); qn=ik.step(Tt,q,dt); trav+=np.abs(qn-q); q=qn
        p,r=err(F.fk(q),Tgoal)
        if conv is None and k*dt>=ramp_s and p<0.01 and r<5: conv=k*dt
    p,r=err(F.fk(q),Tgoal); return dict(travel=np.degrees(trav), maxdev=np.degrees(np.abs(q-q0)), p=p*100, r=r, conv=conv)

if __name__=="__main__":
    ev=events(); print(len(ev),"events")
    variants={
      "kit 30Hz cap1.5":(dict(),30),
      "quest_collect 60Hz cap2.5 d1e-2":(dict(cap=2.5,damping=1e-2),60),
      "posture engage w.1":(dict(cap=[2,2,2,4,4,4],damping=1e-2,posture=[0,0,0,.1,.1,.1]),60),
      "posture engage w.3":(dict(cap=[2,2,2,4,4,4],damping=1e-2,posture=[0,0,0,.3,.3,.3]),60),
      "posture leaky w.3":(dict(cap=[2,2,2,4,4,4],damping=1e-2,posture=[0,0,0,.3,.3,.3],anchor="leaky"),60),
      "posture engage all .05/.3":(dict(cap=[2,2,2,4,4,4],damping=1e-2,posture=[.05,.05,.05,.3,.3,.3]),60),
    }
    for name,(kw,hz) in variants.items():
        rows=[run(IK(**kw),q0,F.fk(q1),hz) for _,q0,q1 in ev]
        wt=np.array([r["travel"][3]+r["travel"][5] for r in rows]); P=np.array([r["p"] for r in rows]); R=np.array([r["r"] for r in rows])
        flips=(np.array([max(r["maxdev"][3],r["maxdev"][5]) for r in rows])>90).sum()
        conv=[r["conv"] for r in rows if r["conv"] is not None]
        print(f"{name:34s} wrist travel med {np.median(wt):5.0f} max {wt.max():5.0f} deg | j4/j6 >90deg: {flips:2d}/{len(rows)} | final err med {np.median(P):.1f}cm {np.median(R):.1f}deg, max {P.max():.1f}cm {R.max():.0f}deg | converged {len(conv)}/{len(rows)} med {np.median(conv) if conv else float('nan'):.2f}s")
