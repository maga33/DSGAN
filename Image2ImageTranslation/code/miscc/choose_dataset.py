'''Choose the dataset according to the name information'''
def choose_dataset(dataset_name):
    if dataset_name == 'cityscapes':
        from .datasets_city import Dataset
        return Dataset
    elif dataset_name == 'handbags' or dataset_name == "shoes":
        from .datasets_handbags import Dataset
        return Dataset
    elif dataset_name == 'shoes_ori':
        from .datasets_shoes_ori import  Dataset
        return Dataset
    elif dataset_name == 'nyu':
        from .datasets_nyu import Dataset
        return Dataset
    elif dataset_name == 'night2day':
        from .datasets_night2day import Dataset
        return Dataset
    elif dataset_name == 'flowers':
        from .datasets_flowers import Dataset
        return Dataset
    elif dataset_name == 'maps':
        from .datasets_maps import Dataset
        return Dataset
    elif dataset_name == 'ADE20K':
        from .datasets_ADE20K import Dataset
        return Dataset
    elif dataset_name == 'facades':
        from .datasets_facades import Dataset
        return Dataset
