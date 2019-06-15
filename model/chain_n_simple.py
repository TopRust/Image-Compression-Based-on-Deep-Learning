
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
size = \
10
class Chain(chainer.Chain):

    def __init__(self):
        super(Chain, self).__init__()
        coder = []
        decoder = []
        # mask = []
        # 第一层网络
        coder.append(('conv{}'.format(0), L.Convolution2D(3, 9, 5, stride=1, pad=0)))
        # coder.append(('bn{}'.format(0), L.BatchNormalization(9)))
        # coder.append(('_sigmoid{}'.format(0), F.leaky_relu))
        coder.append(('_sigmoid{}'.format(0), F.sigmoid))
        

        # 卷积网络
        for i in range(size - 1):
            coder.append(('conv{}'.format(i + 1), L.Convolution2D(9, 9, 5, stride=1, pad=0)))
            # coder.append(('bn{}'.format(i + 1), L.BatchNormalization(9)))
            # coder.append(('_sigmoid{}'.format(i + 1), F.leaky_relu))
            coder.append(('_sigmoid{}'.format(i + 1), F.sigmoid))
            

        # 反卷积网络
        for i in reversed(range(size - 1)):
            decoder.append(('deconv{}'.format(i + 1), L.Deconvolution2D(9, 9, 5, stride=1, pad=0)))
            # decoder.append(('bn{}'.format(2 * size - i - 2), L.BatchNormalization(9)))
            # decoder.append(('_sigmoid{}'.format(2 * size - i - 2), F.leaky_relu))
            decoder.append(('_sigmoid{}'.format(2 * size - i - 2), F.sigmoid))

        # for i in reversed(range(size - 1)):
            # mask.append(('mdeconv{}'.format(i + 1), L.Deconvolution2D(9, 9, 3, stride=1, pad=0)))
            # mask.append(('bn{}'.format(2 * size - i - 2), L.BatchNormalization(9)))
            # mask.append(('_sigmoid{}'.format(2 * size - i - 2), F.leaky_relu))
            # mask.append(('_msigmoid{}'.format(2 * size - i - 2), F.sigmoid))
                        


        #　最后一层网络
        decoder.append(('deconv{}'.format(0), L.Deconvolution2D(9, 3, 5, stride=1, pad=0)))
        # decoder.append(('bn{}'.format(2 * size - 1), L.BatchNormalization(3)))
        # decoder.append(('_sigmoid{}'.format(2 * size - 1), F.leaky_relu))
        decoder.append(('_sigmoid{}'.format(2 * size - 1), F.sigmoid))

        # mask.append(('mdeconv{}'.format(0), L.Deconvolution2D(9, 3, 3, stride=1, pad=0)))
        # mask.append(('bn{}'.format(2 * size - 1), L.BatchNormalization(3)))
        # mask.append(('_sigmoid{}'.format(2 * size - 1), F.leaky_relu))
        # mask.append(('_msigmoid{}'.format(2 * size - 1), F.sigmoid))


        # 初始化
        with self.init_scope():
            for n in coder:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])
            for n in decoder:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])
            # for n in mask:
                # if not n[0].startswith('_'):
                    # setattr(self, n[0], n[1])

        self.coder = coder
        self.decoder = decoder
        # self.mask = mask

    def forward(self, x):
        for n, f in self.coder:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        # x_mask = Variable(x.array)

        for n, f in self.decoder:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        # for n, f in self.mask:
            # if not n.startswith('_'):
                # x_mask = getattr(self, n)(x_mask)
            # else:
                # x_mask = f(x_mask)
        return x
        # return x, x_mask

model = Chain()
