import sys, numpy as np, pandas as pd, pathlib, mujoco
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import fk as F
LO=np.array([-2.79,-3.14,0,-2.145,-1.482,-2.529])+0.05; HI=np.array([2.79,0,3.14,1.413,1.482,2.529])-0.05
def rotang(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
def cond(q):
    F.D.qpos[:6]=q; mujoco.mj_kinematics(F.M,F.D); mujoco.mj_comPos(F.M,F.D)
    jp=np.zeros((3,F.M.nv)); jr=np.zeros((3,F.M.nv)); mujoco.mj_jacSite(F.M,F.D,jp,jr,F.M.site("tcp").id)
    J=np.vstack([jp,jr*0.1])[:, :6]; s=np.linalg.svd(J,compute_uv=False); return s[0]/s[-1]
for ds in sys.argv[1:]:
  ds=pathlib.Path(ds)
  for f in sorted(ds.glob("data/chunk-000/*.parquet")):
    df=pd.read_parquet(f); a=np.stack(df["action"]).astype(float)
    for side,off in (("R",7),("L",0)):
        A=a[:,off:off+6]
        if np.ptp(A,axis=0).max()<0.05: continue
        dA=np.abs(np.diff(A,axis=0)); sat=(dA>=0.049).any(1)
        i=0;n=len(sat)
        while i<n:
            if not sat[i]: i+=1; continue
            j=i
            while j<n and sat[j]: j+=1
            if j-i>=8:
                T0,T1=F.fk(A[i]),F.fk(A[j]); sweep=A[j]-A[i]
                atlim=[k for k in range(6) if A[i:j+1,k].min()<=LO[k]+0.01 or A[i:j+1,k].max()>=HI[k]-0.01]
                print(f"{ds.name}/{f.stem[-3:]} {side} t={i/30:5.1f}s dur={(j-i)/30:.2f}s sweep(deg)={np.round(np.degrees(sweep)).astype(int)} "
                      f"tcp_move={np.linalg.norm(T1[:3,3]-T0[:3,3])*100:4.1f}cm tcp_rot={rotang(T1[:3,:3]@T0[:3,:3].T):5.1f}deg "
                      f"cond {cond(A[i]):.0f}->{cond(A[j]):.0f} at_limit={atlim}")
            i=j
