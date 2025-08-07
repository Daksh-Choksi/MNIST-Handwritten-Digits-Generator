import gradio as gr
from Generator import Generator
import torch
import torchvision.utils as vutils
import random
from PIL import Image
import numpy as np
import torchvision.transforms as T

def Generate_images(progress=gr.Progress()):
  img_list = []
  gen = Generator(100, 64)
  gen.load_state_dict(torch.load("Generator.pth", map_location=torch.device('cpu')))
  noise = torch.randn(size=(128,100,1,1))
  with torch.inference_mode():
     fake = gen(noise).detach().cpu()
  img_list.append(fake)
  choice = random.choice(random.choice(img_list))
  Transform = T.ToPILImage()
  image = Transform(choice)
  return image

demo = gr.Interface(Generate_images, None, 'image')

demo.launch()

