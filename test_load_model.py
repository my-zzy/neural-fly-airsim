import torch
from mlmodel import load_model

final_model = load_model(modelname = "neural-fly_dim-a-4_v-q-pwm-epoch-199")
# print('Final model loaded:', final_model)
print()
output = final_model.phi(torch.zeros(11))
print(output)

phi = output.unsqueeze(0).repeat(3, 1)
print(phi)