import argparse
import os
import pathlib

from utils import plot_images


def create_image_sample(image_path: str, save_path: str, n: int = 16, n_cols: int = 4):
    images = []
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(pathlib.Path(image_path).iterdir()):
        if image.is_file():
            images.append(image)
        if i >= n - 1:
            break
    fig = plot_images(images, n_cols)
    fig.tight_layout()
    fig.savefig(save_path + "/samples.png")
    print(f"Image saved to {os.path.abspath(save_path + "/samples.png")}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates an image in n_cols x n // n_cols")
    parser.add_argument("filepath", type=str)
    parser.add_argument("-o", "--output", default=".", type=str)
    parser.add_argument("-n", "--n_images", default=16, type=int)
    parser.add_argument("-c", "--n_cols", default=4, type=int)

    args = parser.parse_args()
    create_image_sample(args.filepath, args.output, args.n_images, args.n_cols)

