from jrdb_det3d_eval import eval_jrdb


det_dir = "/home/jupyter-dominik/Person_MinkUNet_adapt/logs/unet_bl_voxel_jrdb_0.05_0.1_EVAL_20231025_145353/output/val/e000000/detections"
# gt_dir = "/globalwork/jia/jrdb_eval/val"
gt_dir = "/scai_data/data01/daav/JRDB/train_dataset/labels_kitti"

ap_dict, ret_dict = eval_jrdb(gt_dir, det_dir)
for k, v in ap_dict.items():
    print(k, v)
