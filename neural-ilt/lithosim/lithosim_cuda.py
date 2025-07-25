import numpy as np
import os, time
import struct
import math
from PIL import Image

import torch
import torch.fft
import torchvision
from torchvision.transforms import Compose, ToTensor, Grayscale
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Torchinductor does not support code generation for complex operators. Performance may be worse than eager.')


def generate_zernike_phase(shape, coeffs, device='cpu'):
    """
    shape: (H, W)
    coeffs: список коэффициентов Цернике [a1, ..., aN_Z], shape (N, N_Z)
    Возвращает фазовую маску φ(u, v) размера (H, W)
    """
    H, W = shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                          torch.linspace(-1, 1, W, device=device), indexing='ij')
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    N = len(coeffs)
    Z = torch.zeros_like(r)[None,:].repeat_interleave(N, 0) # N x H x W
    
    def zernike_polynomial(n, m):
        """Генерирует полином Цернике Z_n^m"""
        mask = (r <= 1).float()

        R = torch.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coef = ((-1)**k * math.factorial(n - k) /
                (math.factorial(k) *
                    math.factorial((n + abs(m))//2 - k) *
                    math.factorial((n - abs(m))//2 - k)))
            R += coef * r**(n - 2*k)

        if m > 0:
            z = R * torch.cos(m * theta)
        elif m < 0:
            z = R * torch.sin(abs(m) * theta)
        else:
            z = R

        return z * mask

    # Пример — вручную добавим несколько низших порядков
    n_m_list = [
        (0,0),    # Z0
        (1,-1),   # Z1
        (1,1),    # Z2
        (2,-2),   # Z3
        (2,0),    # Z4
        (2,2),    # Z5
        (3,-3),   # Z6
        (3,-1),   # Z7
        (3,1),    # Z8
        (3,3),    # Z9
        (4,0),    # Z10
        (4,2),    # Z11
        (4,-2),   # Z12
        (4,4),    # Z13
        (4,-4)    # Z14
    ]

    for i, coeff in enumerate(coeffs.T):
        if i >= len(n_m_list):
            break
        n, m = n_m_list[i]
        Z += coeff[:, None, None] * zernike_polynomial(n, m)

    return Z


def is_bin(filename):
    return any(
        filename.endswith(extension)
        for extension in [".bin"])

def kernels_tensor_preprocess(kernels):
    kernels = kernels.permute(0, 2, 1, 3)
    kernels_real = kernels.select(-1, 0).unsqueeze(-1)
    kernels_imagine = kernels.select(-1, 1).unsqueeze(-1)
    kernels = torch.cat((kernels_real, kernels_imagine), -1)
    kernels = torch.view_as_complex(kernels)
    return kernels

def kernel_bin_preprocess(root_path, kernel_type, kernels_number=24, verbose: bool = True):
    r"""
    Preprocessing SOCS kenerls
    """
    if kernel_type not in ['focus', 'defocus']:
        raise NotImplementedError('Kernel type should be [focus] or [defocus]')
    kernels_filename = ["fh%d.bin" % x for x in range(kernels_number)]
    kernels_path = [os.path.join(root_path, kernel_type + '_data', x)
                    for x in kernels_filename if is_bin(x)]
    kernels_array = []

    for i in range(len(kernels_path)):
        kernels_w = 0
        kernels_h = 0
        with open(kernels_path[i], 'rb') as f:
            data = f.read()
            kernels_w = data[3]
            kernels_h = data[7]
            if verbose:
                print("kernel_%d kernels_w: %d, kernels_h:%d" % (i, kernels_w, kernels_h))
        with open(kernels_path[i], 'rb') as f:
            data = f.read()
            data = data[20:]
            cur_kernel = np.zeros(kernels_w*kernels_h*2)
            for j in range(kernels_h*kernels_w*2):
                single_data = struct.unpack('!f', data[j*4:(j+1)*4])[0]
                if math.isnan(single_data):
                    fill_num = 0.0
                    raise ValueError("current data is %s" % single_data)
                    print("index:%d is nan, filled with %f" % (j, fill_num))
                    single_data = fill_num
                cur_kernel[j] = single_data
            kernels_array.append(cur_kernel)
    total_array = np.asarray(kernels_array)
    reshape_array = np.zeros((kernels_number, 35, 35, 2))

    # For safety reshape
    for kerNum in range(len(kernels_path)):
        for x in range(kernels_w):
            for y in range(kernels_h):
                for a in range(2):
                    offset = x * 35 * 2 + y * 2 + a
                    reshape_array[kerNum, x, y, a] = total_array[kerNum, offset]

    # Generate conjugate transpose of kernels
    reshape_array_ct = np.copy(reshape_array)
    reshape_array_ct[:, :, :, 1] = reshape_array_ct[:, :, :, 1] * -1
    reshape_array_ct = reshape_array_ct.transpose(0, 2, 1, 3)

    # Load in weight
    weight_array = np.zeros(kernels_number)
    weight_path = os.path.join(root_path, kernel_type + '_data', 'scales.txt')
    with open(weight_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                assert kernels_number == int(line)
                continue
            weight_array_idx = idx - 1
            if weight_array_idx >= kernels_number:
                break
            weight_array[weight_array_idx] = float(line)

    # Save as PyTorch Tensor
    kernel_tensor = torch.from_numpy(reshape_array).float()  # K * H * W * 2
    kernel_ct_tensor = torch.from_numpy(reshape_array_ct).float()  # K * H * W * 2
    weight_tensor = torch.from_numpy(weight_array).float()  # K

    kernel_tensor = kernels_tensor_preprocess(kernel_tensor) # K * H * W (complex)
    kernel_ct_tensor = kernels_tensor_preprocess(kernel_ct_tensor) # K * H * W (complex)

    torch.save(kernel_tensor, 'lithosim_kernels/torch_tensor/kernel_' + kernel_type + '_tensor.pt')
    torch.save(kernel_ct_tensor, 'lithosim_kernels/torch_tensor/kernel_ct_' + kernel_type + '_tensor.pt')
    torch.save(weight_tensor, 'lithosim_kernels/torch_tensor/weight_' + kernel_type + '_tensor.pt')

    return kernel_tensor, kernel_ct_tensor, weight_tensor

def load_image(image_path):
    r"""
    Load image and convert to PyTorch Tensor
    """
    image = Image.open(image_path)
    transforms = Compose([
        Grayscale(num_output_channels=1),
        ToTensor(),
    ])
    image = transforms(image)
    return image  # N * 1 * H * W

def tensor_real_to_complex(tensor, dose=1.0):
    r"""
    Convert real tensor to complex tensor, zero is filled in maginary part
    """
    real_tensor = tensor.unsqueeze(-1) * dose
    image_tensor = torch.zeros_like(real_tensor) * dose
    complex_image_data = torch.cat((real_tensor, image_tensor), -1)
    return torch.view_as_complex(complex_image_data) # N * 1 * H * W or 1 * H * W (complex)


def frequency_multiplication(data, kernels):
    r"""
    Multiplication between data and kernels in freq-domain
    """
    assert kernels.dtype == torch.cfloat

    N, ker_num, kernel_height, kernel_width = kernels.shape
    if len(data.shape) == 3: # data.shape is N * 1 * H * W (complex)
        data = data.unsqueeze(1) # N * 1 * H_K * W_K (complex)

    data_width = data.shape[-1]
    data_height = data.shape[-2]
    data_width_half = data_width // 2
    data_height_half = data_height // 2
    x0 = data_width_half - kernel_width // 2
    y0 = data_height_half - kernel_height // 2
    x1 = x0 + kernel_width
    y1 = y0 + kernel_height

    # Data dimension expand to 24
    data = data.repeat_interleave(ker_num, dim=-3)  # N * K * H * W (complex)
    data_new = torch.zeros_like(data)
    
    # Only convolve in the freq-domain image's center
    data_new[..., y0:y1, x0:x1] = data[..., y0:y1, x0:x1] * kernels

    return data_new

def tensor_weight_sum(data, weight, square_root=False, normalized_weight=False):
    r"""
    Convert complex data to real data and do weighted sum
    """
    assert data.size(-3) == weight.size(0)

    if square_root == True:
        squeeze_data = data.abs()
    else:
        squeeze_data = data.abs() ** 2 # reduce last dimension(real+imagine)

    if len(squeeze_data.shape) == 3:  # squeeze_data K * H * W (real)
        weight = weight.reshape(-1, 1, 1)
    elif len(squeeze_data.shape) == 4:  # squeeze_data N * K * H * W (real)
        weight = weight.reshape(1, -1, 1, 1)
    else:
        raise NotImplementedError("squeeze_data should be [K * H * W] or [N * K * H * W]")

    if normalized_weight == True:
        return (squeeze_data * weight).sum(dim=-3, keepdim=True) / weight.sum()
    else:
        return (squeeze_data * weight).sum(dim=-3, keepdim=True) # return tensor's shape is N * 1 * H * W (real)

def mask_threshold(intensity_map, threshold):
    r"""
    Intensity map to binary wafer
    """
    return (intensity_map >= threshold).float()

def lithosim(image_data, threshold, kernels, weight, wafer_output_path, save_bin_wafer_image,
             kernels_number=None, avgpool_size=None, dose=1.0, return_binary_wafer=True,
             zernike_coeffs=None):
    r"""
    Lithography simulation main function
    Args:
        image_data: mask image
        threshold: constant threshold for resist model 
        kernels: SOCS kernels 
        weight: weight of SOCS kernels
        dose: +-2% simulation dose \in {0.98, 1.0, 1.02}
    Outputs:
        tensors, intensity image and binary wafer image
    """
    if kernels_number is not None:
        if kernels_number == 0:
            kernels = tensor_real_to_complex(torch.ones((1,35,35), device=kernels.device))
            weight = torch.ones((1,), device=weight.device)
        else:
            kernels = kernels[:kernels_number]
            weight = weight[:kernels_number]
    complex_image_data = tensor_real_to_complex(image_data, dose=dose) # N * 1 * H * W (complex)
    complex_image_data = fft2(complex_image_data) # N * 1 * H * W (complex)
    
    kernels = kernels.unsqueeze(0) # 1 x K x H x W

    # ======= фазовая маска через Цернике ========
    if zernike_coeffs is not None:
        if len(zernike_coeffs.shape) == 1:
            zernike_coeffs = zernike_coeffs.unsqueeze(0) # 1 x N_Z

        _, K, H, W = kernels.shape
        device = kernels.device
        phase = generate_zernike_phase((H, W), zernike_coeffs, device=device)
        phase_mask = torch.exp(1j * phase)  # N x H x W (complex)
        phase_mask = phase_mask.unsqueeze(1)  # N x 1 x H x W
        kernels = kernels * phase_mask  # broadcasting to N x K x H x W
    # ============================================

    complex_image_data = frequency_multiplication(complex_image_data, kernels) # N * K * H * W (complex)
    complex_image_data = ifft2(complex_image_data) # N * K * H * W (complex)
    intensity_map = tensor_weight_sum(complex_image_data, weight) # N * 1 * H * W (real)

    if avgpool_size is not None:
        avg_layer = torch.nn.AvgPool2d(
            kernel_size=(avgpool_size, avgpool_size), 
            stride=(avgpool_size, avgpool_size))
        intensity_map = avg_layer(intensity_map) # N * 1 * (H / avgpool_size) * (W / avgpool_size)

    binary_wafer = None  # If return_binary_wafer == False, can save GPU memory
    if return_binary_wafer:
        binary_wafer = mask_threshold(intensity_map, threshold)

    if save_bin_wafer_image == True:
        torchvision.utils.save_image(binary_wafer, wafer_output_path)
    return intensity_map, binary_wafer

def convolve_kernel(image_data, kernels, weight, dose=1, combo_kernel=True):
    r"""
    Calculation of convolve(image_data, kernels)
    Args:
        image_data: mask image
        kernels: SOCS kernels 
        weight: weight of SOCS kernels
        dose: +-2% process condition \in {0.98, 1.0, 1.02} -> min/nomial/max
        combo_kernel: pre-computed kernel combinations for acceleration, from MOSAIC (GAO et al., DAC'14)
    Outputs:
        tensors, convolved complex image data
    """
    if image_data.dtype == torch.cfloat: # If image data is already in complex format, convolveCpxKernel()
        complex_image_data = image_data # N * 1 * H * W (complex)
    else: # Transform real image data to complex image data, convolveMaskKenerl()
        complex_image_data = tensor_real_to_complex(image_data, dose) # N * 1 * H * W (complex)
    
    complex_image_data = fft2(complex_image_data) 
    if combo_kernel == True:
        complex_image_data = frequency_multiplication_combo(complex_image_data, kernels, weight) # N * 1 * H * W (complex)
    else:
        complex_image_data = frequency_multiplication(complex_image_data, kernels) # N * K * H * W (complex)

    complex_image_data = ifft2(complex_image_data)

    # NOTE: weight for gradient should be square root of litho weight
    if combo_kernel == False:
        assert complex_image_data.size(-3) == weight.size(0)
        # Weight sum in channel dimension
        weight = weight.sqrt()
        if len(complex_image_data.shape) == 3: # complex_image_data is K * H * W (complex)
            weight = weight.reshape(-1, 1, 1)
        elif len(complex_image_data.shape) == 4: # complex_image_data is N * K * H * W (complex)
            weight = weight.reshape(1, -1, 1, 1)
        else:
            raise NotImplementedError("complex_image_data should be [K * H * W] or [N * K * H * W]")

        complex_image_data = (complex_image_data * weight).sum(dim=-3, keepdim=True)

    return complex_image_data # N * 1 * H * W (complex)

@torch.compile
def frequency_multiplication_combo(data, kernels, weight):
    r"""
    Pre-computed kernel combinations for acceleration, from MOSAIC (GAO et al., DAC'14)
    """
    weight = weight.sqrt().reshape(-1, 1, 1)
    kernels_combo = (weight * kernels).sum(dim=-3, keepdim=True)
    return frequency_multiplication(data, kernels_combo)


# FFT functions (From FacebookAIResearch)
# https://github.com/facebookresearch/fastMRI/blob/master/banding_removal/fastmri/data/transforms.py
def fft2(data):
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=[-1,-2])
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data

def ifft2(data):
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifftn(data, dim=[-1,-2])
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data
