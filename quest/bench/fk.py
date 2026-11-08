import re, numpy as np, mujoco, pathlib
HERE = pathlib.Path(__file__).parent
def model():
    text = (HERE/"metal.urdf").read_text()
    text = re.sub(r"<(visual|collision)\b.*?</\1>", "", text, flags=re.DOTALL)
    comp = '<mujoco><compiler discardvisual="true" fusestatic="false" balanceinertia="true" boundmass="0.001" boundinertia="0.000001"/></mujoco>'
    text = re.sub(r"(<robot\b[^>]*>)", rf"\1{comp}", text, count=1)
    spec = mujoco.MjSpec.from_string(text); spec.body("link6").add_site(name="tcp", pos=[0.135,0,0])
    return spec.compile()
M = model(); D = mujoco.MjData(M)
def fk(q):
    D.qpos[:6] = q; mujoco.mj_kinematics(M, D)
    s = M.site("tcp").id
    T = np.eye(4); T[:3,:3] = D.site_xmat[s].reshape(3,3); T[:3,3] = D.site_xpos[s]; return T
