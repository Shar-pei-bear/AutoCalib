syms B11 B22 B13 B23 B33 hi1 hi2 hi3 hj1 hj2 hj3 real
B = [B11, 0, B13; 0, B22, B23; B13, B23, B33];
hi = [hi1; hi2; hi3];
hj = [hj1; hj2; hj3];

f = simplify(expand(hi.'*B*hj));
V = jacobian(f,[B11 B22 B13 B23 B33]);

v = -B23/B22;
u = -B13/B11;

lambda = (B13*B13 + v*(-B11*B23))/B11;
lambda = simplify(expand(lambda))
