import torch
import torch.nn as nn
from torch.nn import functional as F
from numba import cuda
import torch
import yaml
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperParameters():
    pass

# -----------------------------------------------------------------------------
# Helper functions
def validate_config(object, required_keys):
    for key, expected_type in required_keys.items():
        if not hasattr(object, key):
            raise ValueError(f"Missing required configuration key: {key}")
        value = getattr(object, key)
        if not isinstance(value, expected_type):
            # Log the type of each key
            logger.error(f"{key}: {type(value).__name__} (expected: {expected_type.__name__})")
            raise TypeError(f"Expected {key} to be of type {expected_type.__name__}, got {type(value).__name__}")

def process_yaml_to_class(yaml_data: dict, class_instance: object, prefix: str = ''):
    for key, value in yaml_data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            process_yaml_to_class(value, class_instance, prefix=f"{prefix}_{key}" if prefix else key)
        else:
            # Set attribute based on the key name without dictionaries
            setattr(class_instance, f"{prefix}_{key}" if prefix else key, value)
            # logger.info(f"Added attribute: {prefix}_{key} = {value}")

def load_yaml_to_class(file_path: str, class_instance: object):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        process_yaml_to_class(data, class_instance)
# -----------------------------------------------------------------------------

class DeviceConfig():
    def __init__(self):
        self.device_type = "None"
        self.device = "None"
        self.has_ampere_device= False
        self._check_ampere_generation()
        self._auto_detect_cuda_device()
        # added after video, pytorch can be serious about it's device vs. device_type distinction
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

    def _check_ampere_generation(self):
        ampere_devices = []
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            device = torch.cuda.get_device_properties(i)
            compute_capability = device.major
            if compute_capability == 8:  # Major version 8 indicates Ampere generation
                ampere_devices.append(device.name)

        if ampere_devices:
            self.has_ampere_device =True
            logger.info("Ampere generation CUDA devices found:")
            for device_name in ampere_devices:
                logger.info(f" - {device_name}")
        else:

            logger.info("No Ampere generation CUDA devices found.")

    def _auto_detect_cuda_device(self):
        # Autodetect CUDA device
        if torch.cuda.is_available():
            self.device ="cuda"
        elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            self.device ="mps"
        else:
            self.device ="cpu"

        logger.info(f"Using device {self.device}")

class TrainConfig():
    def __init__(self,file_path:str):
        self.file_path = file_path
        self.config=HyperParameters()
        self.device=DeviceConfig()
        load_yaml_to_class(file_path, self.config)
        if (self.config.train_mixed_precision_training_enable and self.device.has_ampere_device):
            self.config.dtype= self.configure_mp_dtype(self.config.train_mixed_precision_training_dtype)
            logger.info(f"Assigning dtype to {self.config.dtype}")
        else:
            self.config.dtype= torch.float32
            logger.warning(f"Assigning dtype to {self.config.dtype}")

        # Print all attributes
        logger.info("Attributes of HyperParameters class:")
        for key in dir(self.config):
            if not key.startswith('_') and not callable(getattr(self.config, key)):
                logger.info(f"{key}: {getattr(self.config, key)}")

    # def _validate_config(self):
    #     required_keys = {
    #         'optimizer': dict,
    #     }

    #     validate_config(self.config,required_keys)

    #     # Validate required attributes
    #     self._validate_config()

    def get_lr(self,it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.optimizer_params_warmup_steps:
            return self.config.optimizer_params_max_lr * (it+1) / self.config.optimizer_params_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.train_max_steps:
            return self.config.optimizer_params_max_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.optimizer_params_warmup_steps) / (self.config.train_max_steps - self.config.optimizer_params_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.config.optimizer_params_max_lr + coeff * (self.config.optimizer_params_max_lr - self.config.optimizer_params_max_lr)

    def configure_mp_dtype(self,dtype_string):
        """
        Sets the dtype of a given variable based on the dtype_string.

        Args:
            variable (torch.Tensor): The variable to be cast to the new dtype.
            dtype_string (str): The desired dtype as a string. Should be one of 'FP32', 'FP16', 'BF16', 'TF32'.

        Returns:
            torch.Tensor: The variable cast to the new dtype.
        """
        dtype_string = dtype_string.upper()
        if dtype_string == 'FP32':
            return (torch.float32)
        elif dtype_string == 'FP16':
            return (torch.float16)
        elif dtype_string == 'BF16':
            return (torch.bfloat16)
        elif dtype_string == 'TF32':
            # TF32 is not a native dtype in PyTorch; it's a hardware setting.
            # Operations using TF32 will still be in FP32 but with TF32 settings applied in CUDA.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return (torch.float32)
        else:
            raise ValueError(f"Unsupported dtype string: {dtype_string}")




if __name__=="__main__":
    # Usage example
    file_path = 'train_config.yaml'
    processor = TrainConfig(file_path)



