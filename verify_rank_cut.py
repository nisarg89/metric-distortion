"""Polynomial certification for direct charging to random rank cuts.
This verifies fixed stronger crossing residuals and exact polynomial cut laws
while reusing unchanged, previously verified polynomial bounds.
Every cutoff is read from the fixed rational witness; no root search is run.
"""
import finite_family_verifier as f
from fractions import Fraction as F
from pathlib import Path
import json, hashlib, time
from certificate_paths import CODE, DATA, OUTPUT, load_data
ROOT=OUTPUT
psq=lambda x:(f.law.mu-sum(w*x**int(d) for w,d in zip(f.law.weights,f.law.degrees)))/f.law.den
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((ROOT/'finite-family-verification.json').read_text())
assert old['status']=='verified'
assert sha(ROOT/'finite-family-witness.json')==old['witness_sha256']
assert sha(CODE/'finite_family_verifier.py')==old['checker_sha256']
fixed=load_data('rank-cut-witness.json')
rows=[];laws=[];total=0;st=time.monotonic()
for E in f.es:
    h=F(fixed['cut_functions'][f.es.index(E)]['cutoff'])
    assert f.b<h<1
    delta=-(h+f.k*E*psq(h))
    assert delta>0
    eta=f.k*E/f.law.den
    co={0:f.C-delta-eta*f.law.mu,1:F(-1)}
    for w,d in zip(f.law.weights,f.law.degrees):co[int(d)]=co.get(int(d),F(0))+eta*w
    value=lambda a:sum(v*a**i for i,v in co.items())
    assert value(h)==f.C and value(f.b)>0
    # Nonnegative derivative of the middle polynomial and upward initial jump.
    def neg_derivative(z):
        a=z[0]
        out=f.Jet(1)
        for w,d in zip(f.law.weights,f.law.degrees):out-=eta*w*d*f.jpow(a,int(d)-1)
        return out
    der=f.check(neg_derivative,((f.b,h),(F(0),F(0)),(F(0),F(0)),(F(0),F(0))),limit=100000)
    # Stronger version of every original crossing polynomial.
    checks=[]
    for i,cell in enumerate(f.cells(E)):
        base=f.fun('high-internal-cross',cell,E)
        mode=cell[2]
        def shifted(z,base=base,mode=mode):
            p=z[0]
            return base(z)+delta*(p*(1-p) if mode=='middle' else 1)
        ans=f.check(shifted,((cell[0],cell[1]),(F(0),F(1)),(F(0),F(1)),(F(0),F(1))),limit=600000)
        checks.append({'cell':i,'p0':str(cell[0]),'p1':str(cell[1]),**ans});total+=ans['nodes']
    (ROOT/'rank-cut-partial.json').write_text(json.dumps({'rows':rows,'laws':laws,'nodes':total},indent=2)+'\n')
    rows.append({'E':str(E),'h':str(h),'delta':str(delta),'negative_derivative':der,'crossing_checks':checks})
    laws.append({'physical_E':str(E),'cutoff':str(h),'downward_shift':str(delta),'middle_coefficients':{str(i):str(v) for i,v in co.items()},'pieces':[{'interval':['0',str(f.b)],'formula':'0','right_endpoint_included':True},{'interval':[str(f.b),str(h)],'formula':'listed middle polynomial','left_endpoint_included':False,'right_endpoint_included':True},{'interval':[str(h),'1'],'formula':str(f.C),'left_endpoint_included':False}], 'upward_jump_at_b':str(value(f.b))})
    print(json.dumps({'E':str(E),'nodes':total,'seconds':time.monotonic()-st}),flush=True)
witness=json.loads((ROOT/'finite-family-witness.json').read_text())
witness['cut_functions']=laws
witness['cut_distribution_definition']='F_j(z) is C times Pr(T_j<z); T_j is supported on [0,1]. At optimal-candidate position r use min(T_j,r).'
witness['admissible_root_sets']=[{'family':jj['index'],'intervals':[[jj['r0'],jj['r1']],[str(f.b),'1']]} for jj in witness['root_intervals']]
(ROOT/'rank-cut-witness.json').write_text(json.dumps(witness,indent=2)+'\n')
verification={'status':'cut_polynomials_verified_pending_algebraic_composition','reused_original_verification_sha256':sha(ROOT/'finite-family-verification.json'),'reused_witness_sha256':old['witness_sha256'],'added_polynomial_boxes':total,'seconds':time.monotonic()-st,'rows':rows,'witness_sha256':sha(ROOT/'rank-cut-witness.json')}
(ROOT/'rank-cut-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
