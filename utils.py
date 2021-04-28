

def set_device(args):
    if args.cpu:
        if args.gpu is not None:
            raise ArgumentError("Can't use CPU and GPU at the same time")
    elif args.gpu is None:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def removeStateDictPrefix(state_dict, len=6):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[len:]
        new_state_dict[name] = v
    return new_state_dict