import lithosim_cuda as litho
import os, argparse
import torch
import torch.utils.checkpoint as cp

    
class LTSimulator(torch.nn.Module):
    def __init__(self, device: str = None, checkpointing: bool = False):
        super().__init__()
        parser = argparse.ArgumentParser(description='take parameters')
        parser.add_argument('--kernels_root', type=str,
                            default='lithosim_kernels/bin_data')
        parser.add_argument('--kernel_type', type=str,
                            default='focus', help='[focus] or [defocus]')
        parser.add_argument('--input_root', type=str, default='../dataset/input/')
        parser.add_argument('--output_root', type=str,
                            default='../output/litho_output')
        parser.add_argument('--kernel_num', type=int, default=24, help='24 SOCS kernels')
        parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
        parser.add_argument('--threshold', type=float, default=0.225, help='Resist threshold')
        # args = parser.parse_args()
        self.config = parser.parse_known_args()[0]
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        kernels, weights = self.load_kernels_weights()
        self.kernels = kernels
        # print(torch.std(self.kernels))
        self.weights = weights
        
        self.checkpointing = checkpointing
    
    def load_kernels_weights(self):
        kernels_root = self.config.kernels_root
        _, _, _ = litho.kernel_bin_preprocess(kernels_root, 'focus', verbose=False)
        _, _, _ = litho.kernel_bin_preprocess(kernels_root, 'defocus', verbose=False)

        litho_kernel_type = self.config.kernel_type
        torch_data_path = 'lithosim_kernels/torch_tensor'
        output_root = self.config.output_root

        if not os.path.exists(output_root):
            os.makedirs(output_root)

        kernels_path = os.path.join(
            torch_data_path, 'kernel_' + litho_kernel_type + '_tensor.pt')
        weight_path = os.path.join(
            torch_data_path, 'weight_' + litho_kernel_type + '_tensor.pt')

        kernels = torch.load(kernels_path, map_location=self.device)
        weights = torch.load(weight_path, map_location=self.device)
        return kernels, weights
    
    def run_lithosim(self, image_data, kernels=None, weights=None, zernike_coeffs=None):
        r"""
        Run lithography simulation for a batch of masks (within the arg.input_root folder)
        """

        save_bin_wafer_image = False
        kernel_number = self.config.kernel_num
        threshold = self.config.threshold
        save_name = None
        if kernels is None or weights is None:
            kernels = self.kernels.clone().to(device=image_data.device)
            weights = self.weights.clone().to(device=image_data.device)
        # print('kernels', image_data.device, torch.std(self.kernels))
        
        # runner = checkpoint(litho.lithosim, use_reentrant=False) if self.checkpointing else litho.lithosim
        if self.checkpointing:
            def run_func(x):
                return litho.lithosim(x, threshold, kernels, weights, save_name, save_bin_wafer_image,
                                        kernel_number, zernike_coeffs=zernike_coeffs)
            intensity_map, binary_wafer = cp.checkpoint(run_func, image_data, use_reentrant=False)
        else:
            intensity_map, binary_wafer =  litho.lithosim(image_data, threshold, kernels, weights, save_name, save_bin_wafer_image,
                                        kernel_number, zernike_coeffs=zernike_coeffs)

        # intensity_map, binary_wafer = litho.lithosim(image_data, threshold, kernels, weights, save_name, save_bin_wafer_image,
        #                                 kernel_number, zernike_coeffs=zernike_coeffs)
        return intensity_map#.squeeze(1) # N x H x W
