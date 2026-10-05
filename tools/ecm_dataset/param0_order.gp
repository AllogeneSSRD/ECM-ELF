\\ Suyama PARAM=0 model shared with src/core/ecm_driver.cpp's --go helper.
\\ b*y^2=x^3+A*x^2+x, base point (x,1).  X=b*x,Y=b^2*y
\\ gives E=[0,b*A,0,b^2,0].  We compute both #E and the point order.
ecm_param0_order(p,s) = {
  my(u,v,x,A,b,E,P,n,fn,o,fo);
  if(!isprime(p),error("factor must be a proven prime"));
  u=Mod(s^2-5,p);v=Mod(4*s,p);
  if(u==0 || v==0,error("singular Suyama initialization"));
  x=u^3;A=(3*u+v)*(v-u)^3/(4*x*v)-2;x=x/v^3;
  b=x*(x*(x+A)+1);
  if(b==0 || A^2==4,error("singular curve"));
  E=ellinit([0,b*A,0,b^2,0]);P=[b*x,b^2];
  if(!ellisoncurve(E,P),error("base point mapping failed"));
  n=ellcard(E);fn=factor(n);o=ellorder(E,P,n);fo=factor(o);
  for(i=1,matsize(fn)[1],if(!isprime(fn[i,1]),error("unproven group factor")));
  for(i=1,matsize(fo)[1],if(!isprime(fo[i,1]),error("unproven point factor")));
  if(n%o || ellmul(E,P,o)!=[0],error("invalid point order"));
  for(i=1,matsize(fo)[1],if(ellmul(E,P,o/fo[i,1])==[0],error("point order is not minimal")));
  print("GROUP|",n);print("POINT|",o);
  for(i=1,matsize(fn)[1],print("G|",fn[i,1],"|",fn[i,2]));
  for(i=1,matsize(fo)[1],print("P|",fo[i,1],"|",fo[i,2]));
  print("OK");
};
ecm_factor(n) = {
  my(f=factor(n));
  for(i=1,matsize(f)[1],
    if(!isprime(f[i,1]),error("unproven factor"));
    print("F|",f[i,1],"|",f[i,2]));
  print("OK");
};
