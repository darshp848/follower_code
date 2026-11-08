import sys,pathlib; sys.path.insert(0,str(pathlib.Path(__file__).parent))
import numpy as np, collections, mink
import bench_ik as B, fk as F
def ang(R): return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
class GuardIK(B.IK):
    def __init__(s, window_s=0.5, min_travel_deg=30, ratio=0.3, **kw):
        super().__init__(**kw); s.window_s=window_s; s.min_travel=min_travel_deg; s.ratio=ratio; s.engage(None)
    def engage(s,q):
        super().engage(q) if q is not None else None
        s.off=np.eye(3); s.win=collections.deque(); s.rebases=0
    def step(s,target,q,dt):
        t=target.copy(); t[:3,:3]=s.off@target[:3,:3]
        qn=super().step(t,q,dt)
        R0,R1=F.fk(q)[:3,:3],F.fk(qn)[:3,:3]
        s.win.append((float(np.degrees(np.abs(qn[3:]-np.asarray(q)[3:]).sum())), ang(R1@R0.T)))
        while len(s.win)>max(1,int(s.window_s/dt)): s.win.popleft()
        tr=sum(w for w,_ in s.win); rot=sum(r for _,r in s.win)
        if tr>s.min_travel and rot<s.ratio*tr:
            # self-motion: the wrist turns a lot for little tool turn. Hold the tool's orientation from here.
            s.off=R0@target[:3,:3].T; s.win.clear(); s.rebases+=1
            t[:3,:3]=R0; qn=super().step(t,q,dt)
        return qn
if __name__=="__main__":
    ev=B.events()
    for name,kw,hz in (("kit",dict(),30),("guard 60Hz cap2.5",dict(cap=2.5,damping=1e-2),60),("guard 60Hz wrist4",dict(cap=[2,2,2,4,4,4],damping=1e-2),60),
                       ("guard r.25 wrist4",dict(cap=[2,2,2,4,4,4],damping=1e-2,ratio=0.25),60)):
        rows=[];reb=0
        for _,q0,q1 in ev:
            ik=GuardIK(**kw) if name!="kit" else B.IK(); r=B.run(ik,q0,F.fk(q1),hz); rows.append(r); reb+=getattr(ik,"rebases",0)
        wt=np.array([r["travel"][3]+r["travel"][5] for r in rows]); P=np.array([r["p"] for r in rows]); R=np.array([r["r"] for r in rows])
        flips=(np.array([max(r["maxdev"][3],r["maxdev"][5]) for r in rows])>90).sum()
        print(f"{name:20s} wrist travel med {np.median(wt):4.0f} max {wt.max():4.0f} | >90deg {flips}/{len(rows)} | pos err med {np.median(P):.2f} max {P.max():.2f} cm | rot err med {np.median(R):.0f} max {R.max():.0f} deg | rebases {reb}")
