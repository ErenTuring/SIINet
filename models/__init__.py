from models.HF_FCN import HF_FCN
from models.DeeplabV3_plus import DeeplabV3_plus
from models.ContrastNets import ResUnet, ResUnet_SIIS
from models.SIIS_NET import (
    Vgg_SIIS, Resnet_SIIS, Deeplab_SIIS
)
from models.UNet import UNet, UNet_SIIS


def build_model(num_classes, siis_size=None, net_num='12345678'):
    '''
    Args:
        net_num: A string of eight numbers,
        The first number represents the arch of the model:
            if net_num[0] <= 5
                1 - /
                2 - UNet_SIIS
                3 - ResUnet_SIIS
                4 - deeplabv3+
                The second number represents the arch of SIIS
                    X0 - no SIIS
                    X1 - SIIS1 exp
                    X2 - SIIS2 exp
                    X3 - SIIS3 exp
                    X4 - SIIS4 (Conv3d-RNN)
                    X5 - SIIS5 exp
                    X6 - SIIS6 exp
                    X7 - SIIS7 (conv1d)
                The third number represents the width of SIIS:
                    XXn - width=n
                The fourth number represents the kw of SIIS:
                    XXXn - kw = [3, 5, 7, 9, 11, 13](n)
                The fifth number represents the dim of SIIS:
                    XXXXn - dim = [128, 256](n)
                if backbone of model is resnt:
                    The sixth number represents the arch of resnet:
                        XXXXXn - resnet_arch = ['resnet50', 'resnet101'](n)
                    The seventh number represents the layer_num of resnet:
                        XXXXXXn - layer_num = n
                    The eighth number represents the out_stride of resnet:
                        XXXXXXXn - out_stride = [8, 16, 32](n)
            elif net_num[0] > 5:
                6 - depplabV3+
                    The second number represents the scheme of ASPP_Rate:
                    The third number represents the arch of resnet:
                        XXn - resnet_arch = ['resnet50', 'resnet101'](n)
                    The fourth number represents the layer_num of resnet:
                        XXXn - layer_num = n
                    The fifth number represents the out_stride of resnet:
                        XXXXn - out_stride = [8, 16, 32](n)
                7 - ResUnet
                8 - HF_FCN
                9 - UNet
    '''
    assert type(net_num) == str
    paras = [int(n) for n in net_num]  # str -> int
    model_name = {
        # 1: 'Vgg_SIIS', 2: 'Resnet_scnIIS
        1: 'DinkNet34', 2: 'UNet_SIIS',
        3: 'ResUnet_SIIS', 4: 'Deeplab_SIIS',
        6: 'DeeplabV3_plus', 7: 'ResUnet', 8: 'HF_FCN', 9: 'UNet'
    }

    print('-'*50, '\n', 'Net_num: ', net_num)
    commen_str = model_name[paras[0]] + '(' + 'num_classes'

    # 解析net_num
    if paras[0] < 6:
        # Nets with SIIS
        if siis_size is not None:
            commen_str += ', siis_size'
        commen_str += ', arch=%d' % paras[1]
        commen_str += ', width=%d' % paras[2]
        commen_str += ', kw=%d' % [3, 5, 7, 9, 11, 13][paras[3]]
        commen_str += ', dim=%d' % [128, 256, 512][paras[4]]
        if paras[0] > 3:  # use resnet as basic encoder
            commen_str += ', resnet_arch=%s' % ["'resnet50'", "'resnet101'"][paras[5]]
            commen_str += ', layer_num=%d' % paras[6]
            commen_str += ', output_stride=%d' % [8, 16, 32][paras[7]]
    elif paras[0] == 6:
        commen_str += ', aspp_rate=%d' % paras[1]
        commen_str += ', resnet_arch=%s' % ["'resnet50'", "'resnet101'"][paras[2]]
        commen_str += ', layer_num=%d' % paras[3]
        commen_str += ', output_stride=%d' % [8, 16, 32][paras[4]]

    commen_str += ')'
    print(commen_str)
    model = eval(commen_str)  # Build model

    return model
