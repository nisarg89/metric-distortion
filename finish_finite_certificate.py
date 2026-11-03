"""Complete the polynomial checks and emit paper-normalized witness data."""
import finite_family_verifier as f
from fractions import Fraction as F
from pathlib import Path
import json,hashlib,bisect
from certificate_paths import CODE, DATA, OUTPUT, load_data
ROOT=OUTPUT

def scalar_check(fun,lo,hi):
 return f.check(fun,((lo,hi),(F(0),F(0)),(F(0),F(0)),(F(0),F(0))),limit=500000)

# All extra statements are direct polynomial interval bounds.
extra={'psi_nonnegative_below_cut':scalar_check(lambda z:-f.jpsi(f.law,z[0]),F(0),f.b),'potentials':[],'endpoints':[]}
for E in f.es:
 ans=scalar_check(lambda z:z[0]+f.k*E*f.jpsi(f.law,z[0])-f.C,f.b,F(1))
 extra['potentials'].append({'E':str(E),'nonnegative_high_potential':ans})
 # Full-mass first bounds: low prefixes and high prefixes.
 def lowfirst(z):
  u,v=z[:2];a=f.b*u;r=f.mix(f.b,1,v)
  return a+f.k*E*f.jpsi(f.law,r)-f.C
 box=((F(0),F(1)),(F(0),F(1)),(F(0),F(0)),(F(0),F(0)))
 ef={'E':str(E),'p0_first':{'polynomial':str(-f.C),'upper':float(-f.C)},
     'p0_internal_low':{'polynomial':'0','upper':0},
     'p0_internal_high':ans,
     'p1_first_low_prefix':f.check(lowfirst,box,limit=500000),
     'p1_first_high_prefix':{'method':'exact polynomial identity and nonpositive monomial coefficients','identity':'eta*(psi(r)-psi(a)) = -eta*(r-a)*sum_d w_d sum_i r^(d-1-i) a^i/(1-mu)','nonnegative_difference_domain':'0 <= a <= r <= 1','upper':0}}
 for low in [True,False]:
  def fullinternal(z):
   a=f.b*z[0] if low else f.mix(f.b,1,z[0])
   wa=f.Jet(0) if low else f.C-a-f.k*E*f.jpsi(f.law,a)
   w1=f.C-1+f.k*E # normalized psi(1)=-1
   return a-f.k*E+wa-w1
  ef['p1_internal_'+('low_prefix' if low else 'high_prefix')]=scalar_check(fullinternal,F(0),F(1))
 for r0,r1,ee in f.bands:
  if E==ee:
   ef['p1_first_low_root']=scalar_check(lambda z:z[0]+f.k*E*f.jpsi(f.law,z[0])-f.C,r0,r1)
 extra['endpoints'].append(ef)
# Verify all coefficient identities used for the factorized polynomial evaluations.
import sympy as sp
p,u,cap=sp.symbols('p u A');cv=lambda q:sp.Rational(q.numerator,q.denominator)
for degree in [1,2,32]:
 left=cap-p*(1-u)
 # The pure monomial identity for average psi minus psi(cap).
 ss=sum((degree-i)*cap**(degree-1-i)*left**i for i in range(degree))
 expr=(degree+1)*cap**degree-sum(cap**(degree-i)*left**i for i in range(degree+1))-(cap-left)*ss
 assert sp.Poly(sp.expand(expr),cap,p,u).is_zero
 # H(a)/p = u*(H(a)/a) at a=pu is a polynomial identity.
 assert sp.expand((p*u)**(degree+1)-p*u*(p*u)**degree)==0
x=sp.symbols('x')
hp=lambda xx:(cv(f.law.mu)*xx-sum(cv(w)*xx**(int(d)+1)/(int(d)+1) for w,d in zip(f.law.weights,f.law.degrees)))/cv(f.law.den)
Bp=sum(cv(w)/((int(d)+1)*cv(f.law.den))*sum(x**i for i in range(1,int(d)+1)) for w,d in zip(f.law.weights,f.law.degrees))
assert sp.Poly(sp.expand(hp(x)-(1-x)*Bp),x).is_zero
extra['factorization_identities']='verified exactly for degrees 1, 2, 32'

# Read the completed full-domain run, with no root/E analytical reductions.
run=ROOT/('finite-family-all-'+'-'.join(str(e).replace('/','_') for e in f.es)+'-0-100000.json')
direct=json.loads(run.read_text());assert direct['coefficients']==list(map(str,f.es)) and direct['group']=='all'
for E in f.es:
 rows=[r for r in direct['rows'] if r['E']==str(E)];cc=f.cells(E)
 assert len(rows)==len(cc)
 assert all(r['cell']==i and F(r['p0'])==cc[i][0] and F(r['p1'])==cc[i][1] for i,r in enumerate(rows))
 assert all({(c['kind'],tuple(c['root_interval'])) for c in row['checks']}=={(kind,(str(lo),str(hi))) for kind,lo,hi in f.tasks(E,cc[i])} for i,row in enumerate(rows))
assert f.bands[0][0]==0 and f.bands[-1][1]==f.b
assert all(r0<r1 for r0,r1,E in f.bands)
assert all(hi==lo for (a,hi,E),(lo,c,ee) in zip(f.bands,f.bands[1:]))

# Polynomial data in the paper's normalization psi=mu-g.
def add(a,b):
 return [ (a[i] if i<len(a) else F(0))+(b[i] if i<len(b) else F(0)) for i in range(max(len(a),len(b))) ]
def scale(a,c):return [v*c for v in a]
def mul(a,b):
 out=[F(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):out[i+j]+=x*y
 return out
end=F(1)-f.law.mu;eta=[f.k*E/end for E in f.es]
allcells={E:f.cells(E) for E in f.es};knots=sorted({x for cc in allcells.values() for cell in cc for x in cell[:2]});mass=[]
for lo,hi in zip(knots,knots[1:]):
 mid=(lo+hi)/2;mode,bc,bp,tc,tp=f.rawprice(lo,hi)
 if mode=='small':beta=[F(0),4*f.k,-4*f.k];theta=[f.k,4*f.k]
 elif mode=='large':beta=[F(0),f.k];theta=[f.k]
 else:beta=[bc,bp];theta=[tc,tp]
 nums=[]
 for E in f.es:
  cc=allcells[E];i=bisect.bisect_right([c[0] for c in cc],mid)-1;cell=cc[i];assert cell[0]<=lo<hi<=cell[1]
  branch=cell[-1]
  if branch==0:num=[F(0)]
  elif branch==1:num=mul(add(theta,[-f.k*E]),[F(0),F(1),F(-1)])
  else:num=mul(add([F(0),f.k*E],scale(beta,-1)),[F(1),F(-1)])
  nums.append(list(map(str,scale(num,1/end))))
 mass.append({'p0':str(lo),'p1':str(hi),'lambda_numerator':list(map(str,scale(beta,1/end))),'charge_numerators':nums})
witness={'status':'verified','g_degrees':list(map(str,f.law.degrees)),'g_weights':list(map(str,f.law.weights)),
 'normalization':'psi(x)=mu-g(x), H(x)=integral_0^x psi','mu':str(f.law.mu),'C':str(f.C),'b':str(f.b),
 'function_definition':'On p0 <= p < p1, divide the listed numerator polynomial by p*(1-p); polynomial coefficients are in ascending powers of p.',
 'eta':list(map(str,eta)),'endpoints':{'lambda(0)':'0','lambda(1)':'0','t_j(0)':'0','t_j(1)':'eta_j'},
 'potential':'W_j(a)=0 for a<=b, C-a-eta_j*psi(a) for a>b',
 'root_intervals':[{'r0':str(lo),'r1':str(hi),'index':f.es.index(E)} for lo,hi,E in f.bands],
 'root_domain':'For r in root interval J_j use j. For b <= r <= 1 all indices work.',
 'mass_cells':mass}
assert witness==load_data('finite-family-witness.json'), 'Generated coefficients differ from the frozen witness'
(ROOT/'finite-family-witness.json').write_text(json.dumps(witness,indent=2)+'\n')
summary={'status':'verified','scope':'All local inequalities on full proper-mass and endpoint domains; nine finite choices; rational cut b; no analytic root/E/endpoint reduction assumed.',
 'direct_poly_boxes':direct['nodes'],'direct_mass_cells':len(direct['rows']),'common_polynomial_mass_cells':len(mass),
 'witness_sha256':hashlib.sha256((ROOT/'finite-family-witness.json').read_bytes()).hexdigest(),
 'checker_sha256':hashlib.sha256((CODE/'finite_family_verifier.py').read_bytes()).hexdigest(),
 'extra_polynomial_checks':extra}
(ROOT/'finite-family-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({kk:vv for kk,vv in summary.items() if kk!='extra_polynomial_checks'},indent=2),flush=True)
