import resp as R, numpy as np
ORIG=R.PoseFilter
class PF2:
    def __init__(s,pmin,pbeta,rmin,rbeta,d=1.0):
        s.p=R.OneEuro(pmin,pbeta,d); s.rmin,s.rbeta,s.d=rmin,rbeta,d; s.rot=None; s.w=0.0
    def __call__(s,T,dt):
        o=np.eye(4); o[:3,3]=s.p(T[:3,3],dt)
        if s.rot is None: s.rot=T[:3,:3].copy()
        else:
            ax,an=R.axis_angle(T[:3,:3]@s.rot.T); s.w+=R.OneEuro._alpha(s.d,dt)*(an/dt-s.w)
            a=R.OneEuro._alpha(s.rmin+s.rbeta*s.w,dt); s.rot=R.rotation_about(ax,a*an)@s.rot
        o[:3,:3]=s.rot; return o
