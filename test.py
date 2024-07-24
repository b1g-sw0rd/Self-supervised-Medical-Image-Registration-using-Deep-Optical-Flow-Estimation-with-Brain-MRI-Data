import torch
import torchvision.transforms as transforms


# 假设你有一个单通道的图像张量image，形状为 [H, W]
image = torch.randn( 256, 256)  # 示例，随机生成一个单通道图像

# 创建一个转换函数来将图像张量转换为PIL图像
to_pil = transforms.ToPILImage()

# 将图像张量转换为PIL图像
pil_image = to_pil(image)

# 保存PIL图像
pil_image.save("single_channel_image.png")