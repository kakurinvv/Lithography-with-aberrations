import lithosim_cuda as litho
import os
from fnmatch import fnmatch
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from lt_simulator import LTSimulator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class PNGFolderDataset(Dataset):
    def __init__(self, root, transform=None, image_size=224, device: str = 'cpu',
                 N_zernike: int = 6, aberr_max: float = 1., load_all_to_ram: bool = False,
                 num_threads: int = 4):
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
        self.N_zernike = N_zernike
        self.aberr_max = aberr_max
        self.device = device
        self.load_all_to_ram = load_all_to_ram
        self.num_threads = num_threads if num_threads > 0 else os.cpu_count()

        # Кэш для design_img и litho_img
        self.design_images_in_ram = None
        self.litho_images_in_ram = None

        if self.load_all_to_ram:
            self._cache_all_images()

    def _process_single_image(self, img_path):
        """Загружает и готовит design_img и litho_img для одного файла."""
        design_img = litho.load_image(img_path).to(self.device, non_blocking=True)
        if self.transform is not None:
            design_img = self.transform(design_img)
        litho_img = self.sim.run_lithosim(design_img)
        return design_img.cpu(), litho_img.cpu()

    def _cache_all_images(self):
        """Параллельное кэширование всех картинок с прогрессбаром."""
        self.design_images_in_ram = [None] * len(self.filenames)
        self.litho_images_in_ram = [None] * len(self.filenames)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_idx = {
                executor.submit(self._process_single_image, path): idx
                for idx, path in enumerate(self.filenames)
            }

            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Caching images"):
                idx = future_to_idx[future]
                design_img, litho_img = future.result()
                self.design_images_in_ram[idx] = design_img.cpu()
                self.litho_images_in_ram[idx] = litho_img.cpu()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Используем RAM, если включено
        if self.load_all_to_ram and self.design_images_in_ram is not None:
            design_img = self.design_images_in_ram[idx].to(self.device, non_blocking=True)
            litho_img = self.litho_images_in_ram[idx].to(self.device, non_blocking=True)
        else:
            img_path = self.filenames[idx]
            design_img, litho_img = self._process_single_image(img_path)

        # Генерируем аберрации
        # zernike_coeffs = (torch.rand(self.N_zernike, device=litho_img.device) - 0.5) * 2 * self.aberr_max
        zernike_coeffs = torch.randn(self.N_zernike, device=litho_img.device) * self.aberr_max / 3
        amp = torch.abs(zernike_coeffs).max()
        zernike_coeffs = (zernike_coeffs / amp)**5 * amp
        aberrated_img = self.sim.run_lithosim(design_img, zernike_coeffs=zernike_coeffs)

        imgs = torch.cat([design_img.unsqueeze(1), litho_img, aberrated_img], dim=0).squeeze(1)  # 3 x H x W
        return imgs, zernike_coeffs

    def intensity_tensor_to_image(self, intensity):
        return tensor_to_image(litho.mask_threshold(intensity, self.sim.config.threshold))


def tensor_to_image(tensor):
    if tensor.ndim == 3:
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return tensor.detach().cpu().numpy()
