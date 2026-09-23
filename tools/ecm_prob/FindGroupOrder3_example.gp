FindGroupOrder3(p, s) = {
    \\ 建议显式地将 2^32 作为 Mod(p) 处理，以确保进行模逆运算
    A = Mod(4*s, p) / Mod(2^32, p) - 2;
    
    b = 4*A + 10;

    \\ 初始化椭圆曲线 y^2 = x^3 + a1*x + a3 ... 这里 a1=0, a2=b*A, a3=0, a4=b^2, a6=0
    E = ellinit([0, b*A, 0, b^2, 0]);
    
    \\ 计算并返回群的阶
    ellcard(E);
}

p = 614002928307599;
s = 1707370477;

order = FindGroupOrder3(p, s);
print(order);
print(factor(order));
