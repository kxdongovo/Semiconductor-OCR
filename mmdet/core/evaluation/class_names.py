import mmcv


def wider_face_classes():
    return ['face']


def voc_classes():
    return [
     '0-0','0-90','0-180','0-270','1-0','1-90','1-180','1-270','2-0','2-90','2-180','2-270','3-0','3-90','3-180','3-270',
'4-0','4-90','4-180','4-270','5-0','5-90','5-180','5-270', '6-0','6-90','6-180','6-270','7-0','7-90','7-180','7-270','8-0','8-90','8-180','8-270',
'9-0','9-90','9-180','9-270','A-0','A-90','A-180','A-270','B-0','B-90','B-180','B-270','C-0','C-90','C-180','C-270','D-0','D-90','D-180','D-270',
'E-0','E-90','E-180','E-270','F-0','F-90','F-180','F-270','G-0','G-90','G-180','G-270','H-0','H-90','H-180','H-270','I-0','I-90','I-180','I-270','J-0',
'J-90','J-180','J-270','K-0','K-90','K-180','K-270','L-0','L-90','L-180','L-270','M-0','M-90','M-180','M-270','N-0','N-90','N-180','N-270',
'O-0','O-90','O-180','O-270','P-0','P-90','P-180','P-270','Q-0','Q-90','Q-180','Q-270','R-0','R-90','R-180','R-270','S-0','S-90','S-180','S-270',
'T-0','T-90','T-180','T-270','U-0','U-90','U-180','U-270','V-0','V-90','V-180','V-270','W-0','W-90','W-180','W-270','X-0','X-90','X-180','X-270',
'Y-0','Y-90','Y-180','Y-270','Z-0','Z-90','Z-180','Z-270'
    ]


def imagenet_det_classes():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def imagenet_vid_classes():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def coco_classes():
    return [
         '0-0','0-90','0-180','0-270','1-0','1-90','1-180','1-270','2-0','2-90','2-180','2-270','3-0','3-90','3-180','3-270',
'4-0','4-90','4-180','4-270','5-0','5-90','5-180','5-270', '6-0','6-90','6-180','6-270','7-0','7-90','7-180','7-270','8-0','8-90','8-180','8-270',
'9-0','9-90','9-180','9-270','A-0','A-90','A-180','A-270','B-0','B-90','B-180','B-270','C-0','C-90','C-180','C-270','D-0','D-90','D-180','D-270',
'E-0','E-90','E-180','E-270','F-0','F-90','F-180','F-270','G-0','G-90','G-180','G-270','H-0','H-90','H-180','H-270','I-0','I-90','I-180','I-270','J-0',
'J-90','J-180','J-270','K-0','K-90','K-180','K-270','L-0','L-90','L-180','L-270','M-0','M-90','M-180','M-270','N-0','N-90','N-180','N-270',
'O-0','O-90','O-180','O-270','P-0','P-90','P-180','P-270','Q-0','Q-90','Q-180','Q-270','R-0','R-90','R-180','R-270','S-0','S-90','S-180','S-270',
'T-0','T-90','T-180','T-270','U-0','U-90','U-180','U-270','V-0','V-90','V-180','V-270','W-0','W-90','W-180','W-270','X-0','X-90','X-180','X-270',
'Y-0','Y-90','Y-180','Y-270','Z-0','Z-90','Z-180','Z-270'
    ]


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace'],
    'cityscapes': ['cityscapes']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
