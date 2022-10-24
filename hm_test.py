import habitat
import habitat_sim

def main():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/gxh-pc/data/scene_datasets/hm3d/train/00009-vLpv2VX547B/vLpv2VX547B.basis.glb"
    backend_cfg.scene_dataset_config_file = "/home/gxh-pc/data/scene_datasets/hm3d/train/hm3d_annotated_train_basis.scene_dataset_config.json"

if __name__ == '__main__':
    main()