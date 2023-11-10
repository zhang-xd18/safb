import os
import random
import thop
import torch

from models import fbnet, renet, jnet
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device

def create_model(L, cr, mode):
    if mode == 'FB':
        model = fbnet(L, cr)
    elif mode == 'RE': 
        model = renet()
    elif mode == 'Joint':
        model = jnet(L, cr)
    else:
        raise ValueError   
    return model

def init_model(args):
    # Model loading
    model = create_model(L=args.L, cr=args.cr, mode=args.mode)
    
    if args.pretrained is not None: 
        assert os.path.isfile(args.pretrained)       
        if args.pretrained2 is None:
            logger.info("loading checkpoint from one model")
            state_dict = torch.load(args.pretrained,
                                    map_location=torch.device('cpu'))['state_dict']
            # update manually
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

            logger.info("pretrained model loaded from {}".format(args.pretrained))
        
        else:
            assert os.path.isfile(args.pretrained2)
            logger.info("loading checkpoint from two models seperately")
            FBNet_state_dict = torch.load(os.path.join(args.pretrained),
                                    map_location=torch.device('cpu'))['state_dict']
            RENet_state_dict = torch.load(os.path.join(args.pretrained2),
                                    map_location=torch.device('cpu'))['state_dict']          
            FBNet_model_dict = model.FBNet.state_dict()
            FBNet_state_dict = {k:v for k,v in FBNet_state_dict.items() if k in FBNet_model_dict}
            FBNet_model_dict.update(FBNet_state_dict)
            model.FBNet.load_state_dict(FBNet_model_dict)
            RENet_model_dict = model.RENet.state_dict()
            RENet_state_dict = {k:v for k,v in RENet_state_dict.items() if k in RENet_model_dict}
            RENet_model_dict.update(RENet_state_dict)
            model.RENet.load_state_dict(RENet_model_dict)

            logger.info("pretrained model loaded from {} and {}".format(args.pretrained, args.pretrained2))

    # Model flops and params counting
    if args.mode == 'Joint':
        image1 = torch.randn([1, 2, args.L, 32])
        image2 = torch.ceil(torch.rand([1, args.L])*32)
        flops, params = thop.profile(model, inputs=(image1,image2,), verbose=False)
    elif args.mode == 'FB':
        image = torch.randn([1, 2, args.L, 32])
        flops, params = thop.profile(model, inputs=(image,), verbose=False)
    elif args.mode == 'RE':
        image = torch.randn([1, 2, 32, 32])
        flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.2f")

    # Model info logging
    logger.info(f'=> Model [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio={args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
