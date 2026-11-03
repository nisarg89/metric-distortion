"""Independent exact audit of exported rational witness definitions and coverage."""
from fractions import Fraction as F
from pathlib import Path
from bisect import bisect_right
import json,hashlib
from certificate_paths import DATA, OUTPUT, load_data
ROOT=DATA
w=load_data('finite-family-witness.json')
c=load_data('parameters.json')
weights=list(map(F,w['g_weights']));degrees=list(map(int,map(F,w['g_degrees'])));mu=sum(a/(d+1) for a,d in zip(weights,degrees));den=1-mu;k=F(c['kappa'])
assert mu==F(w['mu']) and sum(weights)==1 and degrees==[1,2,32]
assert weights==list(map(F,['19/73','35/146','1/2']))
eta=list(map(F,w['eta']));es=[v*den/k for v in eta];assert all(F(1)<=e<F(3,2) for e in es)
price={F(p):(F(beta),F(T)) for p,beta,T in c['prices']};knots=sorted(price)
s=F(c['s']);ell=F(c['l']);b=F(w['b'])
def poly(cc,p):return sum(F(v)*p**i for i,v in enumerate(cc))
def old(p):
 if p<=s:return 4*k, k*(1+4*p)
 if p>=ell:return k/(1-p),k
 i=bisect_right(knots,p)-1;l,r=knots[i:i+2];u=(p-l)/(r-l)
 beta=(1-u)*price[l][0]+u*price[r][0];T=(1-u)*price[l][1]+u*price[r][1]
 return beta/(p*(1-p)),T
cells=w['mass_cells'];assert F(cells[0]['p0'])==0 and F(cells[-1]['p1'])==1
assert all(F(a['p1'])==F(bb['p0']) for a,bb in zip(cells,cells[1:]))
checked=0
for cell in cells:
 lo,hi=F(cell['p0']),F(cell['p1']);assert lo<hi
 assert len(cell['lambda_numerator'])<=3
 assert len(cell['charge_numerators'])==len(es)
 assert all(len(cc)<=4 for cc in cell['charge_numerators'])
 # Four distinct exact values determine every numerator of degree at most three.
 for num in [1,2,3,4]:
  p=lo+(hi-lo)*F(num,5);L,T=old(p);dd=p*(1-p)
  assert poly(cell['lambda_numerator'],p)==dd*L/den
  for e,cc in zip(es,cell['charge_numerators']):
   charge=max(F(0),T-k*e,k*e-L*(1-p))
   assert poly(cc,p)==dd*charge/den
   assert charge>=0
   checked+=1
root=sorted(w['root_intervals'],key=lambda z:F(z['r0']));assert F(root[0]['r0'])==0 and F(root[-1]['r1'])==b
assert all(F(x['r1'])==F(y['r0']) for x,y in zip(root,root[1:]))
assert sorted(r['index'] for r in root)==list(range(len(es)))
assert w['endpoints']=={'lambda(0)':'0','lambda(1)':'0','t_j(0)':'0','t_j(1)':'eta_j'}
out={'status':'verified','independent_exact_numerator_evaluations':checked,'mass_cells':len(cells),'root_intervals':len(root),'law_and_normalization_exact':True,'degree_bound':3,'witness_sha256':hashlib.sha256((ROOT/'finite-family-witness.json').read_bytes()).hexdigest()}
(OUTPUT/'finite-family-input-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
