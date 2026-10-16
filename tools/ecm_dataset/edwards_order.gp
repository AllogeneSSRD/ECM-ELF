\\ Prime95 sigma_type=0 / Atkin-Morain, matching ecm_edwards_curve.h.
\\ Preserve B for the direct base-point map, without choosing a square root.
\\ Here B=(2*h^2/(h^2-1))^2; a general Edwards model need not have square B.
ecm_edwards_init(p,sigma) = {
  my(W,Q,s,t,alpha,beta,h,d,x,y,A,B,u,v,E,P);
  if(type(p)!="t_INT" || p<=3 || !isprime(p),error("modulus must be a proven prime > 3"));
  if(type(sigma)!="t_INT" || sigma<1 || sigma>2^64-1,error("sigma must be uint64 and nonzero"));
  W=ellinit([-8,-32],p);
  if(#W==0,error("singular auxiliary curve"));
  Q=ellmul(W,[Mod(12,p),Mod(40,p)],sigma);
  if(Q==[0],error("auxiliary point is infinity"));
  s=Q[1];t=Q[2];alpha=(s-9)/(t+s+16);
  beta=2*alpha*(4*alpha+1)/(8*alpha^2-1);h=2*beta-1;
  d=(2*h^2-1)/h^4;
  if(d==0 || d==1,error("singular Edwards curve"));
  x=h*(4*beta-3)/(6*beta-5);
  y=h*(t^2+50*t-2*s^3+27*s^2-104)/((t+3*s-2)*(t+s+16));
  if(x^2+y^2!=1+d*x^2*y^2,error("invalid Edwards base point"));
  A=2*(1+d)/(1-d);B=4/(1-d);
  E=ellinit([0,B*A,0,B^2,0]);
  if(#E==0,error("singular Weierstrass image"));
  if(y==1,P=[0],if(y==-1,P=[Mod(0,p),Mod(0,p)],
    u=(1+y)/(1-y);v=u/x;P=[B*u,B^2*v]));
  if(!ellisoncurve(E,P),error("base point mapping failed"));
  [E,P,[lift(d),lift(x),lift(y),lift(A),lift(B)]];
};

ecm_edwards_order(p,sigma) = {
  my(model=ecm_edwards_init(p,sigma),E,P,n,fn,o,fo,c);
  E=model[1];P=model[2];c=model[3];
  n=ellcard(E);fn=factor(n);o=ellorder(E,P,[n,fn]);fo=factor(o);
  for(i=1,matsize(fn)[1],if(!isprime(fn[i,1]),error("unproven group factor")));
  for(i=1,matsize(fo)[1],if(!isprime(fo[i,1]),error("unproven point factor")));
  if(n%o || ellmul(E,P,o)!=[0],error("invalid point order"));
  for(i=1,matsize(fo)[1],if(ellmul(E,P,o/fo[i,1])==[0],error("point order is not minimal")));
  print("MODEL|edwards-atkin-morain");
  print("D|",c[1]);print("EDWARDS_X|",c[2]);print("EDWARDS_Y|",c[3]);
  print("MONTGOMERY_A|",c[4]);print("MONTGOMERY_B|",c[5]);
  print("GROUP|",n);print("POINT|",o);
  for(i=1,matsize(fn)[1],print("G|",fn[i,1],"|",fn[i,2]));
  for(i=1,matsize(fo)[1],print("P|",fo[i,1],"|",fo[i,2]));
  print("OK");
};
