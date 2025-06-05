import os
import shutil
from PIL import Image

def prepare_dataset(src_folder, dst_folder, open_list, closed_list):
    os.makedirs(os.path.join(dst_folder, 'open'), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, 'closed'), exist_ok=True)
    # Copia imagens de olhos abertos
    for img_path in open_list:
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_folder, 'open', img_name))
    # Copia imagens de olhos fechados
    for img_path in closed_list:
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_folder, 'closed', img_name))
    print('Dataset preparado em:', dst_folder)

if __name__ == '__main__':
    # Exemplo de uso:
    # Liste os caminhos das imagens de olhos abertos e fechados
    open_imgs = [
        'dataset/open/'
        # ...
    ]
    closed_imgs = [
        'dataset/closed/'
        # ...
    ]
    prepare_dataset('dataset', 'dataset', open_imgs, closed_imgs) 