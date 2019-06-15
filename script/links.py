
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import initializers

class Decon_Network(chainer.Chain):

    def __init__(self, exits_bn, activation_function, number_filter_list):

        super(Decon_Network, self).__init__()

        size = len(number_filter_list)
        decoder = []
        # 反卷积网络
        for i in range(1, size):

            decoder.append(('deconv{}'.format(i), L.Deconvolution2D(number_filter_list[i - 1], number_filter_list[i], 3, stride=1, pad=0)))

            if exits_bn:
                decoder.append(('bn{}'.format(i), L.BatchNormalization(number_filter_list[i])))

            decoder.append(('_avfun', activation_function))
                        

        # 初始化
        with self.init_scope():

            for n in decoder:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.decoder = decoder


    def forward(self, x):

        for n, f in self.decoder:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        return x 

class Center_Decon_Network(chainer.Chain):

    def __init__(self, exits_bn, activation_function, number_filter_list):

        super(Center_Decon_Network, self).__init__()

        size = len(number_filter_list)
        decoder = []
        # 反卷积网络
        for i in range(1, size):

            decoder.append(('deconv{}'.format(i), L.Deconvolution2D(number_filter_list[i - 1], number_filter_list[i], 3, stride=1, pad=0)))

            if exits_bn:
                decoder.append(('bn{}'.format(i), L.BatchNormalization(number_filter_list[i])))

            decoder.append(('_avfun', activation_function))
                        

        # 初始化
        with self.init_scope():

            for n in decoder:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.decoder = decoder


    def forward(self, x):

        for n, f in self.decoder:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        return x[:, :, 5: -5, 5: -5]

class Decon_Network_Init(chainer.Chain):

    def __init__(self, exits_bn, activation_function, number_filter_list):

        super(Decon_Network_Init, self).__init__()

        size = len(number_filter_list)
        decoder = []
        # 反卷积网络
        for i in range(1, size):

            decoder.append(('deconv{}'.format(i), L.Deconvolution2D(number_filter_list[i - 1], number_filter_list[i], 3, stride=1, pad=0, initialW=initializers.GlorotNormal)))

            if exits_bn:
                decoder.append(('bn{}'.format(i), L.BatchNormalization(number_filter_list[i])))

            decoder.append(('_avfun', activation_function))
                        

        # 初始化
        with self.init_scope():

            for n in decoder:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.decoder = decoder


    def forward(self, x):

        for n, f in self.decoder:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        return x 

class Center_Decon_Network_Init(chainer.Chain):

    def __init__(self, exits_bn, activation_function, number_filter_list):

        super(Center_Decon_Network_Init, self).__init__()

        size = len(number_filter_list)
        decoder = []
        # 反卷积网络
        for i in range(1, size):

            decoder.append(('deconv{}'.format(i), L.Deconvolution2D(number_filter_list[i - 1], number_filter_list[i], 3, stride=1, pad=0, initialW=initializers.GlorotNormal)))

            if exits_bn:
                decoder.append(('bn{}'.format(i), L.BatchNormalization(number_filter_list[i])))

            decoder.append(('_avfun', activation_function))
                        

        # 初始化
        with self.init_scope():

            for n in decoder:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.decoder = decoder


    def forward(self, x):

        for n, f in self.decoder:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        return x[:, :, 5: -5, 5: -5]


from chainer import report

class Loss_Classifier(chainer.Chain):

    def __init__(self, predictor, lossfun):

        super(Loss_Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
            self.lossfun = lossfun

    def __call__(self, sample, label):

        prediction = self.predictor(sample)
        loss = self.lossfun(prediction, label)

        report({'loss': loss}, self)
        return loss


