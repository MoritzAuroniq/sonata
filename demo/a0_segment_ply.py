import numpy as np
import open3d as o3d
import sonata
import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None


VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0), 1: (174.0, 199.0, 232.0), 2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0), 4: (255.0, 187.0, 120.0), 5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0), 7: (255.0, 152.0, 150.0), 8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0), 10: (148.0, 103.0, 189.0), 11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0), 14: (247.0, 182.0, 210.0), 16: (219.0, 219.0, 141.0),
    24: (255.0, 127.0, 14.0), 28: (158.0, 218.0, 229.0), 33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0), 36: (227.0, 119.0, 194.0), 39: (82.0, 84.0, 163.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


if __name__ == "__main__":
    PLY_PATH = "/home/neura_ai/Downloads/meeting_room.ply"        
    VOXEL_SIZE = 0.04                      # increase if you run out of GPU memory
    OUTPUT_PATH = "/home/neura_ai/Downloads/output_seg.ply"   

    sonata.utils.set_seed(24525867)

    # Load model
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        custom_config = dict(
            enc_patch_size=[512 for _ in range(5)],
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()

    # Load segmentation head
    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])

    # Load transform
    transform = sonata.transform.default()

    # Load your PLY file
    print(f"Loading {PLY_PATH} ...")
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    print(f"  Loaded {len(pcd.points):,} points")

    # Downsample if needed to avoid OOM
    if VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(f"  Downsampled to {len(pcd.points):,} points (voxel_size={VOXEL_SIZE})")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    # Build point dict — colors are 0-1 in open3d, sonata expects 0-255
    point = {
        "coord": np.asarray(pcd.points).astype(np.float32),
        "color": np.asarray(pcd.colors).astype(np.float32) * 255,
        "normal": np.asarray(pcd.normals).astype(np.float32),
    }

    point = transform(point)

    # Inference
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)

        point = model(point)

        # Unpool back to full resolution
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]

    # Print class distribution
    print("\nPredicted class distribution:")
    for i, label in enumerate(CLASS_LABELS_20):
        count = (pred == i).sum()
        if count > 0:
            print(f"  {label}: {count:,} points ({100 * count / len(pred):.1f}%)")

    # Visualize
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    result_pcd.colors = o3d.utility.Vector3dVector(color / 255)

    if OUTPUT_PATH:
        o3d.io.write_point_cloud(OUTPUT_PATH, result_pcd)
        print(f"\nSaved segmented point cloud to {OUTPUT_PATH}")

    o3d.visualization.draw_geometries([result_pcd])