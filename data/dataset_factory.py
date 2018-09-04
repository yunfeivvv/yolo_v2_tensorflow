from data import VOC2007
# from data import VOC2012
# from data import xx
# from data import xxx

datasets_map = {
    'VOC2007': VOC2007,
    # 'VOC2012':VOC2012,
    # 'xx':xx,
    # 'xxx':xxx,
}

def get_data(dataset_name, split_name, dataset_dir, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
        name: String, the name of the dataset.
        split_name: A train/test split name.
        dataset_dir: The directory where the dataset files are stored.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)

    return datasets_map[dataset_name].get_split(split_name,
                                                dataset_dir+dataset_name+'/',
                                                reader)
