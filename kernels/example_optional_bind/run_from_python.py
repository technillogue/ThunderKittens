import micro_add
import torch

input = torch.ones(1, 1, 64, 64, device='cuda')
input2 = torch.ones(1, 1, 64, 64, device='cuda')
print(f"input: {input}")
output = torch.zeros_like(input)
micro_add.add_mats(input, None, output) #output)
#print(f"output: {output}")
print(output.mean(), '\n')
