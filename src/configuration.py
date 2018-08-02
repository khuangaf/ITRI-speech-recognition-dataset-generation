from common import *
import configparser

#
# proposal i,x0,y0,x1,y1,score, label, (aux)
# roi      i,x0,y0,x1,y1
# box        x0,y0,x1,y1



class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version='configuration version \'itri\''

        #features
        self.scales = [  2,  4,  8, 16]
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.iter_accum = 1
        self.min_dim = 128
        self.max_dim = 512
        self.min_scale = 2


    #-------------------------------------------------------------------------------------------------------
    def __repr__(self):
        d = self.__dict__.copy()
        str=''
        for k, v in d.items():
            str +=   '%32s = %s\n' % (k,v)

        return str


    def save(self, file):
        d = self.__dict__.copy()
        config = configparser.ConfigParser()
        config['all'] = d
        with open(file, 'w') as f:
            config.write(f)


    def load(self, file):
        # config = configparser.ConfigParser()
        # config.read(file)
        #
        # d = config['all']
        # self.num_classes     = eval(d['num_classes'])
        # self.multi_num_heads = eval(d['multi_num_heads'])
        raise NotImplementedError
