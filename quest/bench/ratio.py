import sys,pathlib; sys.path.insert(0,str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, fk as F
H=pathlib.Path("C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/nuc_data/hackathon_data")
def ang(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
def windows(ds,col,W=15):
    out=[]
    for f in sorted((H/ds).glob("data/chunk-000/*.parquet")):
        a=np.stack(pd.read_parquet(f)[col]).astype(float)
        for off in (7,0):
            A=a[:,off:off+6]
            if np.ptp(A,axis=0).max()<0.05: continue
            T=[F.fk(q) for q in A]
            for i in range(0,len(A)-W,3):
                wt=np.degrees(np.abs(np.diff(A[i:i+W+1,3:6],axis=0)).sum())
                tr=ang(T[i+W][:3,:3]@T[i][:3,:3].T); tp=np.linalg.norm(T[i+W][:3,3]-T[i][:3,3])*100
                out.append((wt,tr,tp))
    return np.array(out)
for ds,col in (("leader-split-v1-frozen","observation.state"),("plug-unplug-v1","observation.state"),("unplug-v1","action"),("unplug-split-v1","action")):
    w=windows(ds,col); big=w[w[:,0]>45]; r=big[:,1]/big[:,0]
    print(ds,"windows",len(w),"wrist>45deg/0.5s:",len(big),"rot/travel p5,25,50,75:",np.round(np.percentile(r,[5,25,50,75]),2),"frac<0.3:",round(float((r<0.3).mean()),2))
