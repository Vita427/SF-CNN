import scipy.io as sio
import numpy as np
import h5py
import math
import random


data_dir = [r'data\Flevoland_9d.mat',      #400*300
            ]

label_dir =[r'label\Flevoland_label.mat',                #300*400
            ]

FN = 100   # 训练样本数/类
K = 5
class OPT():
    def __init__(self,num=0):
        self.data_dir = data_dir[num]
        self.label_dir = label_dir[num]
        self.samwin =15
        self.rate = 0.02

class Gen_data():
    def __init__(self, data_dir,label_dir,balance = True):
        opt = OPT()
        self.rate = opt.rate
        self.balance = balance
        try:
            load_data = h5py.File(data_dir)
        except Exception as err:
            load_data = sio.loadmat(data_dir)

        try:
            load_label = sio.loadmat(label_dir)
        except Exception as err:
            load_label = h5py.File(label_dir)

        self.all_label = np.array(load_label['label'])
        self.all_data = np.array(load_data['image'])

        # print('all_label.shape',self.all_label.shape)
        # print('all_data.shape',self.all_data.shape)

        #获得图像的宽高 通道 宽度 高度 [c,w,h]-->[h,w,c]
        self.chanel,self.weight,self.height = self.all_data.shape
        self.samwin = opt.samwin
        self.offset = self.samwin // 2
        #对数据进行标准化 并将all_data 进行转换
        self.process()
        # 需要修改 计算出训练集的大小

        self.test_img =  self.all_data[self.offset:-self.offset,self.offset:-self.offset,:]
        self.test_label = self.all_label[self.offset:-self.offset,self.offset:-self.offset]
        self._test_examples = self.test_label.shape[0]*self.test_label.shape[1]

        self.total = np.count_nonzero(self.all_label)
        self.train_rate = int(self.total*opt.rate)
        self.test_num = np.count_nonzero(self.test_label)
        # 训练集和总的样本数目
        # print('train number',self.train_rate)
        # print('tol number',self.total)
        # print('test number',self.test_num)
        # 生成训练集
        self.sample()
        #训练集样本个数
        self._train_examples = self.train_data.shape[0]
        # 样本组
        self.om = self._images
        self.ol = self._labels
        # print("original image",self.om.shape)
        self.shuffle_sample()
        self.one_hot()
        self.omb = self.om
        self.olb = self.ol
        #是否完成一个周期的训练
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.i_step = 0
        self.j_step = 0
        self.step = 1
        self.col = self.weight - self.samwin+1
        self.row = self.height - self.samwin+1
        self.win = self.samwin

    def process(self):
        # print('001.all_data.shape',self.all_data.shape)
        all_datar = np.reshape(self.all_data,[9,self.height*self.weight])
        # for
        mean = np.mean(all_datar, axis=1)
        # print('mean',mean.shape)
        std = np.std(all_datar, axis=1)
        # print('std',std.shape)
        for i in range(self.chanel):
            all_datar[i] = (all_datar[i] - mean[i])/std[i]
        self.all_data = np.reshape(all_datar, [9,self.weight,self.height])
        # print('002.all_data.shape',self.all_data.shape)
        self.all_data = np.transpose(self.all_data,[2,1,0])
        # print('003.all_data.shape',self.all_data.shape)


    def sample(self):
        temp = self.all_label[self.offset:-self.offset,self.offset:-self.offset]
        self.test_img = self.all_data[self.offset:-self.offset,self.offset:-self.offset,:]
        self.test_label = temp
        self._test_examples = temp.shape[0]*temp.shape[1]
        # print('temp.shape',temp.shape)
        # print('self.offset',self.offset)
        labelmax = int(np.max(np.max(temp)))
        self.nclass=labelmax


        # 非平衡样本
        ###################################################
        #                       1
        ###################################################
        if self.balance:
            # print('use balance')
            inr = np.argwhere(temp == 15)
            per = self.train_rate // labelmax
            if inr.shape[0]!= 0 and inr.shape[0]<per:
                per = (self.train_rate - inr.shape[0]//20) // (labelmax-1)

            per = FN
            # print("fix each class num ",per)

            sample = np.zeros([labelmax,per,2])
            # print('sample.shape',sample.shape)
            qie = 0
            for i in range(1,labelmax+1):
                inr = np.argwhere(temp == i)
                # print('every class real num:',i,inr.shape)
                try:
                    ind = random.sample(range(inr.shape[0]),per)
                except Exception :
                    ind = random.sample(range(inr.shape[0]),inr.shape[0]//2)
                    qie = 1

                if qie==1:
                    sample[i - 1, :inr.shape[0]//2, :] = inr[ind]
                else:
                    sample[i-1,:,:]=inr[ind]

            sample = np.reshape(sample, [-1, 2])
            if qie == 1:
                sample = sample[:-(inr.shape[0] - inr.shape[0] // 2), :]
        ###################################################
        #                       2
        ###################################################
        # sample =np.array([])
        #
        # for i in range(1,labelmax+1):
        #     inr = np.argwhere(temp == i)
        #     print('every class real num:',i,inr.shape)
        #     if i==8 or i==15:
        #         per = int(inr.shape[0] * self.rate)
        #         per =25
        #     else:
        #         per =int((6767-50)/15)
        #     ind = random.sample(range(inr.shape[0]), per)
        #     if i ==1:
        #         sample = np.array(inr[ind])
        #     elif i==8 or i==15:
        #         for knx in range(20):
        #             sample=np.concatenate((sample,inr[ind]),axis=0)
        #     else:
        #         sample = np.concatenate((sample, inr[ind]), axis=0)
        # print('sample.shape',sample.shape)
        ###################################################
        #                       3
        ###################################################
        else:
            # print('dont use balance')
            sample =np.array([])
            for i in range(1,labelmax+1):
                inr = np.argwhere(temp == i)
                # print('every class real num:',i,inr.shape)
                # if (i==3 or i==6 or i==8 or i==10 or i==13):
                # if (i == 15):
                #     per = int(inr.shape[0] * self.rate*2)
                # else:
                per = int(inr.shape[0] * self.rate)
                ind = random.sample(range(inr.shape[0]), per)
                if i ==1:
                    sample = np.array(inr[ind])
                else:
                    sample = np.concatenate((sample, inr[ind]), axis=0)

        self.train_data = sample + self.offset

        self._images = np.zeros([self.train_data.shape[0],self.samwin,self.samwin,self.chanel])
        self._labels = np.zeros([self.train_data.shape[0]],dtype='int')

        self.train_data = self.train_data.astype('int')

        for i in range(self.train_data.shape[0]):
            self._labels[i] = self.all_label[self.train_data[i][0],self.train_data[i][1]]
            self._images[i,:,:,:] =self.all_data[self.train_data[i][0]-self.offset:self.train_data[i][0]+self.offset+1,
                                   self.train_data[i][1] - self.offset:self.train_data[i][1] + self.offset + 1,:]


    def shuffle_sample(self):
        perm = np.arange(self._train_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]


    def one_hot(self):
        hotlabel = np.zeros([self._train_examples,self.nclass],dtype='int')

        for i,k in enumerate(self._labels):
            hotlabel[i,k-1] = 1
        self._labels = hotlabel


        hotlabel = np.zeros([self._train_examples, self.nclass], dtype='int')
        for i,k in enumerate(self.ol):
            hotlabel[i,k-1] = 1
        self.ol = hotlabel


    def next_batch(self, batch_size,arguement = True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._train_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._train_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._train_examples
        end = self._index_in_epoch

        if arguement:
            bz = end-start
            inda = np.random.random(bz)>0.5
            tempimages = self._images[start:end]
            #tempimages[inda] = np.flip(tempimages[inda],axis=1)
            rm = np.arange(bz)
            np.random.shuffle(rm)
            tempimages[rm[:bz//4]] = np.rot90(tempimages[rm[:bz//4]],k=1,axes=(1,2))
            tempimages[rm[bz // 4:bz // 4*2]] = np.rot90(tempimages[rm[bz // 4:bz // 4*2]], k=2, axes=(1, 2))
            tempimages[rm[bz // 4*2:bz // 4*3]] = np.rot90(tempimages[rm[bz // 4*2:bz // 4*3]], k=3, axes=(1, 2))

            return tempimages, self._labels[start:end]

        return self._images[start:end], self._labels[start:end]

    def next_batch2(self, batch_size,arguement = True):
        """Return the next `batch_size` examples from this data set."""

        batch_size = batch_size*K

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._train_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._train_examples)
            perm = np.reshape(perm,[-1,K])
            perm = perm.T
            np.random.shuffle(perm)
            perm = perm.T
            np.random.shuffle(perm)
            perm= np.reshape(perm,[-1])

            self.omb = self.om[perm]
            self.olb = self.ol[perm]
            # print("self.om.shape",self.om.shape)
            # print("self.ol.shape", self.ol.shape)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._train_examples
        end = self._index_in_epoch
        return self.omb[start:end], self.olb[start:end]


    def test_data(self):
        return self.test_img

    # def next_test_batch(self, batch_size):
    #     """Return the next `batch_size` examples from this data set."""
    #     start = self._test_flag
    #     self._test_flag += batch_size
    #     if self._test_flag > self._test_examples:
    #         self._test_flag = 0
    #         return self.test_img[start:]
    #     end = self._test_flag
    #     return self.test_img[start:end]

    def next_test_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        # bs = 0
        if self.gettestnum()-(self.j_step +self.i_step* self.col)<batch_size:
            bs= self.gettestnum()-(self.j_step +self.i_step* self.col)
        else:
            bs = batch_size
        data = np.zeros((bs,self.samwin,self.samwin,9))
        # print(self.all_data.shape)
        # print(self.col,self.row)

        for i in range(batch_size):
            data[i, :, :, :] = self.all_data[self.i_step:self.i_step + self.samwin,self.j_step :self.j_step + self.samwin, :]
            self.j_step += 1
            if self.j_step == self.col:
                # print(self.i_step,self.j_step)
                self.j_step = 0
                self.i_step += 1
                if self.i_step == self.row:
                    self.i_step = 0
                    # print(data.shape)
                    return data
        # print(data.shape)
        return data


    def gettestnum(self):
        return self._test_examples



    def all_labels(self):
        return self.all_label

    def get_test(self):
        return self.test_img, self.test_label



    def gen_true(self, h, w, ws, s, temp):
        row = int(math.ceil((h - ws) / s))
        col = int(math.ceil((w - ws) / s))
        label = temp[:row * 3, :col * 3]
        label_true = label.reshape(-1).astype('int')
        return label_true

