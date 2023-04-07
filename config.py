class Config(object):
    def __init__(self):
        self.duration = 50
        self.offset = 0
        
        # training feature
        self.epochs = 100
        
        # data feature
        self.width = 1025
        
        # optimizer parameter
        self.lr = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.weight_decay = 0
        
        