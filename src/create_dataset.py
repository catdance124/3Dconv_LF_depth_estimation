import numpy as np
import pathlib
from PIL import Image
from scripts.PFM_rw import read_pfm


# download full dataset from https://lightfield-analysis.uni-konstanz.de/
full_data_root = pathlib.Path('../full_data')
patch_data_root = pathlib.Path('../patch_data')

def create_EPI_patch(scene_dir, disp):
    stack_v = np.zeros((9, 512, 512, 3), dtype=np.uint8)
    stack_h = np.zeros((9, 512, 512, 3), dtype=np.uint8)
    # vertical
    for i, image_path in enumerate([scene_dir / f'input_Cam{i:03}.png' for i in range(4, 81, 9)]):
        img = np.asarray(Image.open(image_path))
        stack_v[i] = img
    # horizontal
    for i, image_path in enumerate([scene_dir / f'input_Cam{i:03}.png' for i in range(36, 45)]):
        img = np.asarray(Image.open(image_path))
        stack_h[i] = img
    
    # save binary
    save_dir = patch_data_root / scene_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    file_num = 0
    for y in range(0, 38):
        for x in range(0, 38):
            np.save(save_dir / f'{file_num:04}_v.npy', stack_v[:, y*13:y*13+32, x*13:x*13+32])
            np.save(save_dir / f'{file_num:04}_h.npy', stack_h[:, y*13:y*13+32, x*13:x*13+32])
            np.save(save_dir / f'{file_num:04}_disp.npy', disp[   y*13:y*13+32, x*13:x*13+32])
            file_num += 1
    return

def main():
    for disp_path in full_data_root.glob('**/gt_disp_lowres.pfm'):
        # disp pfm -> npy
        disp = read_pfm(disp_path)[0]
        disp = np.flipud(disp)    # y軸反転
        
        # create input patch
        create_EPI_patch(disp_path.parent, disp)
        print(f'done: {disp_path.parent.name}')


if __name__ == "__main__":
    main()