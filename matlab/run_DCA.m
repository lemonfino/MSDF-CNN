dim_drug = 100;
dim_prot = 400;

Q = load('../feature/drug_feature.txt');
nnode1 = size(Q, 1);
alpha1 = 1 / nnode1;
Q1 = log(Q + alpha1) - log(alpha1);
Q2 = Q1 * Q1';
[U1, S1] = svds(Q2, dim_drug);	
X1 = U1 * sqrt(sqrt(S1));

T = load('../feature/protein_feature.txt');
nnode2 = size(T, 1);
alpha2 = 1 / nnode2;
T1 = log(T + alpha2) - log(alpha2);
T2 = T1 * T1';
[U2, S2] = svds(T2, dim_prot);	
X2 = U2 * sqrt(sqrt(S2));

dlmwrite(['../feature/drug_feature100.txt'], X1, '\t');
dlmwrite(['../feature/protein_feature400.txt'], X2, '\t');
