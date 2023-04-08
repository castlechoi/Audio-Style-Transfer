class Config(object):
    def __init__(self):       
        # training feature
        self.epochs = 100
        
        # optimizer parameter
        self.lr = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.weight_decay = 0
        
        # VGG model parameter
        self.in_channel = 1
        self.conv_1_channel = 64
        self.conv_2_channel = 128
        self.conv_3_channel = 256
        self.conv_4_channel = 512
        self.conv_5_channel = 512
        