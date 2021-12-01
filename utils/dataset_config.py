

DATASET_CONFIG = {
    'st2stv2': {
        'num_classes': 174,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'mini_st2stv2': {
        'num_classes': 87,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'st2stv1': {
        'num_classes': 174,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3,
        'label_file': 'something-something-v1-labels.csv'
},
    'kinetics400': {
        'num_classes': 400,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'mini_kinetics400': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'charades': {
        'num_classes': 157,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'diva': {
        'num_classes': 19,
        'train_list_name': 'DIVA_GT_RGB_TSM_train.txt',
        'val_list_name': 'DIVA_GT_RGB_TSM_validate.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:08d}.jpg',
        'filter_video': 0
    },
    'diva_pvi': {
        'num_classes': 8,
        'train_list_name': 'DIVA_PVI_GT_RGB_TSM_train.txt',
        'val_list_name': 'DIVA_PVI_GT_RGB_TSM_validate.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:08d}.jpg',
        'filter_video': 0
    },
    'moments': {
        'num_classes': 339,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mini_moments': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'ucf101': {
        'num_classes': 101,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'hmdb51': {
        'num_classes': 51,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
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
    'activitynet2': {
        'num_classes': 200,
        'train_list_name': 'train_multi.txt',
        'val_list_name': 'val_multi.txt',
        'test_list_name': 'val_multi.txt',
        'filename_seperator': " ",
        'image_tmpl': 'image_{:05d}.jpg',
        'filter_video': 0,
        'label_file': 'categories.txt',
        'multilabel': True
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
