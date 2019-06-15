
import chainer
import chainer.functions as F
import chainer.links as L


class Chain(chainer.Chain):

    def __init__(self):
        super(Chain, self).__init__()

        with self.init_scope():      
            self.conv1=L.Convolution2D(3, 3, 3, stride=1, pad=0)
            self.bn_c1 = L.BatchNormalization(3)
            self.deconv1=L.Deconvolution2D(3, 3, 3, stride=1, pad=0)
            self.bn_d1 = L.BatchNormalization(3)

            self.conv2=L.Convolution2D(3, 3, 3, stride=1, pad=0)
            self.bn_c2 = L.BatchNormalization(3)
            self.deconv2=L.Deconvolution2D(3, 3, 3, stride=1, pad=0)
            self.bn_d2 = L.BatchNormalization(3)

            self.conv3=L.Convolution2D(3, 3, 3, stride=1, pad=0)
            self.bn_c3 = L.BatchNormalization(3)
            self.deconv3=L.Deconvolution2D(3, 3, 3, stride=1, pad=0)
            self.bn_d3 = L.BatchNormalization(3)
    def forward(self, x):
        h = F.relu(self.bn_c1(self.conv1(x)))
        h = F.relu(self.bn_c2(self.conv2(h)))
        h = F.relu(self.bn_c3(self.conv3(h)))

        h = F.relu(self.bn_d3(self.deconv3(h)))
        h = F.relu(self.bn_d2(self.deconv2(h)))
        h = F.relu(self.bn_d1(self.deconv1(h)))

        return h


model = Chain()
