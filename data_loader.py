class dataLoader(object):
    def __init__(self, args):
        self.model = None
        self.data_path = args.data_path
        self.train_batch_size = args.trainBatchSize
        self.val_batch_size = args.valBatchSize
        self.test_batch_size = args.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = args.cuda
        self.x_data = None
        self.y_data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.data_path = args.data_path
        self.train_class_percent = args.train_class_percent
        self.val_class_percent = args.val_class_percent
        self.test_class_percent = 1 - args.train_class_percent - args.val_class_percent
        self.shuffle_list = []