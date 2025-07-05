import lithosim_cuda as litho
import os
from fnmatch import fnmatch
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from lt_simulator import LTSimulator

class PNGFolderDataset(Dataset):
    def __init__(self, root, transform=None, image_size=224, device: str = 'cpu', N_zernike: int=6, aberr_max: float = 1.):
        self.filenames = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, '*.png'):
                    self.filenames.append(os.path.join(path, name))
        self.filenames.sort()
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

        self.sim = LTSimulator(device=device)
        # print(10, torch.std(self.sim.kernels))
        self.N_zernike = N_zernike
        self.aberr_max = aberr_max
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        layout_img = litho.load_image(img_path).to(self.device)
        # print(layout_img.dtype, layout_img)
        if self.transform is not None:
            layout_img = self.transform(layout_img)

        # print(11, torch.std(self.sim.kernels))
        litho_img = self.sim.run_lithosim(layout_img)
        # print(12, torch.std(self.sim.kernels))
        zernike_coeffs = (torch.rand(self.N_zernike, device=litho_img.device) - 0.5) * 2 * self.aberr_max
        # print(13, torch.std(self.sim.kernels))
        aberrated_img = self.sim.run_lithosim(layout_img, zernike_coeffs=zernike_coeffs)
        # print(14, torch.std(self.sim.kernels))

        # imgs = torch.cat([litho_img, aberrated_img], dim=0).squeeze(1) # 2 x H x W
        imgs = torch.cat([layout_img.unsqueeze(1), aberrated_img], dim=0).squeeze(1) # 2 x H x W
        
        return imgs, zernike_coeffs
    
    def intensity_tensor_to_image(self, intensity):
        return tensor_to_image(litho.mask_threshold(intensity, self.sim.config.threshold))

def tensor_to_image(tensor):
    if tensor.ndim == 3:
        return tensor.detach().cpu().numpy().transpose(1,2,0)
    return tensor.detach().cpu().numpy()