class BaseSetting:
    def __init__(self, name):
        self.name = name
        self.train_path = ''
        self.test_path = ''
        self.val_path = ''
        self.bath_size = 8
        self.epoch = 5
    
    def path(self):
        return self.train_path, self.test_path, self.val_path


class Settings:
    def __init__(self):
        self.task = ['SST', 'QA', 'MNLI']
        self.SSTsetting = BaseSetting(name='SST')
        self.QAsetting = BaseSetting(name='QA')
        self.MNLIsetting = BaseSetting(name='MNLI')
        self.bertbaseweight = ''
        self.check_step = 2
        self.learning_rate = 1e-5
        self.num_epoch = 5
    
    def task(self, name):
        if name == 'SST':
            return self.SSTsetting
        elif name == 'QA':
            return self.QAsetting
        else:
            return self.MNLIsetting
