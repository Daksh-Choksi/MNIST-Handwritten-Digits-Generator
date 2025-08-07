from torch import nn

class Generator(nn.Module):
  def __init__(self, input_shape, output_shape):
    super().__init__()
    self.main = nn.Sequential(
      nn.ConvTranspose2d(input_shape, output_shape*16, 4, bias=False),
      nn.BatchNorm2d(output_shape*16),
      nn.ReLU(True),

      nn.ConvTranspose2d(output_shape*16, output_shape*8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(output_shape*8),
      nn.ReLU(True),

      nn.ConvTranspose2d(output_shape*8, output_shape*4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(output_shape*4),
      nn.ReLU(True),

      nn.ConvTranspose2d(output_shape*4, output_shape*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(output_shape*2),
      nn.ReLU(True),

      nn.ConvTranspose2d(output_shape*2, output_shape, 4, 2, 1, bias=False),
      nn.BatchNorm2d(output_shape),
      nn.ReLU(True),

      nn.ConvTranspose2d(output_shape, 1, 4, 1, 1, bias=False),
      nn.Tanh()
    )
  def forward(self,x):
    return self.main(x)