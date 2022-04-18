

DATASET_CONFIG = {
    'mini_kinetics400': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'activitynet': {
        'num_classes': 200,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': 'image_{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    },
    'mini-sports1m': {
        'num_classes': 487,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    },
    'fcvid': {
        'num_classes': 239,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': 'image_{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt'
    }
}


def get_dataset_config(dataset, use_lmdb=False):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['train_list_name']
    val_list_name = ret['val_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['val_list_name']
    test_list_name = ret.get('test_list_name', None)
    if test_list_name is not None:
        test_list_name = test_list_name.replace("txt", "lmdb") if use_lmdb else test_list_name
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)
    multilabel = ret.get('multilabel', False)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file, multilabel
