# 定义种类的名称和对应mask的颜色值，以此来达到分割的目的

"""
RGB color code and object categories:
------------------------------------
000 BW: Background waterbody
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""


import mmcv


def suim_classes():
    """suim class names for external use."""
    return [
        'BW', 'HD', 'PF', 'WR',
        'RO', 'RI', 'FV', 'SR'
    ]

def suim_palette():
    """suim palette for external use."""
    return [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ]


dataset_aliases = {
    'suim': ['suim'],
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


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
