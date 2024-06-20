import torch
device = torch.device('cuda')
pt = torch.load(r"C:\Users\13660\PycharmProjects\OFE-Reg\PWC\pwc_net.pth.tar", map_location=device)
pt['conv1a.0.weight'] = pt['conv1a.0.weight'].sum(1, keepdim=True)