from torch.utils.data import DataLoader
from .dataset_det3d import JRDBDet3D, JRDBDet2D, NuScenesDet3D


def get_dataloader(split, batch_size, num_workers, shuffle, dataset_cfg):
    path = dataset_cfg["path"]
    if dataset_cfg["name"] == "JRDB":
        if dataset_cfg["lidar_type"] == "3D":
            ds = JRDBDet3D(path, split, dataset_cfg)
        elif dataset_cfg["lidar_type"] == "2D":
            ds = JRDBDet2D(path, split, dataset_cfg)
        else:
            raise RuntimeError(f"Unknown lidar_type '{dataset_cfg['lidar_type']}'")
    elif dataset_cfg["name"] == "KITTI":
        ds = JRDBDet3D(path, split, dataset_cfg)
    elif dataset_cfg["name"] == "nuScenes":
        ds = NuScenesDet3D(path, split, dataset_cfg)
    else:
        raise RuntimeError(f"Unknown dataset '{dataset_cfg['name']}'")

    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=ds.collate_batch,
    )
