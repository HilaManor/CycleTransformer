from torch.utils.data import random_split, DataLoader


def get_loaders(args, dataset):
    train_dataset, valid_dataset, test_dataset = get_split_datasets(args, dataset)

    train_loader = DataLoader(train_dataset, batch_size=args["training_args"]["batch_size"], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args["training_args"]["batch_size"], shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args["training_args"]["batch_size"], shuffle=False, drop_last=True)

    return train_loader, valid_loader, test_loader


def get_split_datasets(args, dataset):
    test_db_size = int(args["training_args"]["test_percent"] * len(dataset))
    val_db_size = int(args["training_args"]["val_percent"] * len(dataset))
    train_db_size = len(dataset) - test_db_size - val_db_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_db_size, val_db_size, test_db_size])

    return train_dataset, valid_dataset, test_dataset
