
import chainer
import chainer.functions as F
import chainer.links as L

size = \
5
class Chain(chainer.Chain):

    def __init__(self):
        super(Chain, self).__init__()
        net = []
        # 第一层网络
        net.append(('conv{}'.format(0), L.Convolution2D(3, 9, 5, stride=1, pad=0)))
        net.append(('bn{}'.format(0), L.BatchNormalization(9)))
        net.append(('_sigmoid{}'.format(0), F.sigmoid))

        # 卷积网络
        for i in range(size - 1):
            net.append(('conv{}'.format(i + 1), L.Convolution2D(9, 9, 5, stride=1, pad=0)))
            net.append(('bn{}'.format(i + 1), L.BatchNormalization(9)))
            net.append(('_sigmoid{}'.format(i + 1), F.sigmoid))

        # 反卷积网络
        for i in reversed(range(size - 1)):
            net.append(('deconv{}'.format(i + 1), L.Deconvolution2D(9, 9, 5, stride=1, pad=0)))
            net.append(('bn{}'.format(2 * size - i - 2), L.BatchNormalization(9)))
            net.append(('_sigmoid{}'.format(2 * size - i - 2), F.sigmoid))

        #　最后一层网络
        net.append(('deconv{}'.format(0), L.Deconvolution2D(9, 3, 5, stride=1, pad=0)))
        net.append(('bn{}'.format(2 * size - 1), L.BatchNormalization(3)))
        net.append(('_sigmoid{}'.format(2 * size - 1), F.sigmoid))

        # 初始化
        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])

        self.forwd = net

    def forward(self, x):
        for n, f in self.forwd:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)

        return x
model = Chain()
