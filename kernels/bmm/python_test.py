import torch
import thunderkittens

b, N, K, M = 2, 4096, 4096, 4096
A = torch.rand([b, N, K], dtype=torch.bfloat16, device="cuda")
A_ = A.clone()
B = torch.rand([b, M, K], dtype=torch.bfloat16, device="cuda")
B_ = B.clone()
c = thunderkittens.batch_matmul(A, B)
torch.cuda.synchronize()
d = thunderkittens.batch_matmul(c, B)
torch.cuda.synchronize()
assert A.equal(A_)
ref = A@(B.transpose(-2, -1))
print(c[0, 0,0])
assert ref.allclose(c)

