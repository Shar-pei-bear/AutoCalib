syms a b u v
A = [a, 0, u; 0, b, v; 0, 0, 1];
B = inv(A).'*inv(A);
B = simplify(expand(B));
f = (B(1, 3)*B(1, 3) + v*(-B(1,1)*B(2,3)))/B(1,1);
f = simplify(f);
simplify(expand(B(3, 3) - f))