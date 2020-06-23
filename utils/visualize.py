# -*- coding:utf-8 -*-
import visdom
import time
import numpy as np
# import cv2


class Visualizer(object):
    ''' 封装了visdom的基本操作, 以便快捷可视化.
    启动:
        python -m visdom.server
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}  # 画的第几个数，相当于横座标, 保存（’loss',23） 即loss的第23个点
        self.log_text = ''
        self.class_table = None  # 本次任务的类别表

    def reinit(self, env='default', **kwargs):
        ''' 修改visdom的配置 '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个在不同的坐标轴（pannel）上
        Args:
            d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, extra_opts=None, **kwargs):
        ''' Plot y in the pane with specific name.
        Agrs:
            name - Name of pane
            y - Single number a tuple of number to plot at the same pane
            extra_opts - other opts about plot
                legend: if y is a tuple of number, the corresponding legend needs to be passed in as a dict.
        Examples:
            self.plot('loss', 0.89)
            self.plot('Val evalutions', (0.90, 0.91), dict(legend=['kappa', 'Over_Acc'])
        '''
        # TODO: y must be tup?
        y = np.column_stack(y) if type(y) is tuple else y
        y = y if type(y) is np.ndarray else np.array([y])

        opts = dict(title=name)
        if extra_opts is not None:
            if 'legend' in extra_opts:
                extra_opts.update(dict(showlegend=True))
            opts.update(extra_opts)

        x = self.index.get(name, 0)
        self.vis.line(Y=y, X=np.array([x]),
                      win=name,
                      opts=opts,
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        if not isinstance(img_, np.ndarray):
            img_ = img_.cpu().numpy()
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text', if_print=False):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        if if_print:
            print(info)

        info = ('[{time}] {info} <br>'.format(
            time=time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())),
            info=info))
        self.log_text += info.replace('\n', '<br>')
        self.vis.text(self.log_text, win)
        # self.vis.text(self.log_text, win)

    def plot_hist(self, hist, epoch):
        ''' Confusion Matrix '''
        form = '<table border="1">'
        # category name:
        form += '<tr>'
        c_names = []
        for name, num in self.class_table.items():
            form += '<td>%s</td>' % name
            c_names.append(name)
        # form += '<td>Confusion Matrix</td>'
        form += '<td>Epoch = %d</td>' % epoch
        form += '</tr>'
        # hist
        for i in range(hist.shape[0]):
            form += '<tr>'
            for j in range(hist.shape[1]):
                form += '<td>%d</td>' % hist[i, j]
            form += '<td>%s</td>' % c_names[i]
            form += '</tr>'
        form += '</table>'

        # self.index['Hist'] = x + 1
        self.vis.text(form, win='Confusion Matrix')

    def __getattr__(self, name):
        return getattr(self.vis, name)
