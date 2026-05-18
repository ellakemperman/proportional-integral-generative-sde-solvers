import argparse
import enum
import os.path

import torch

from pi_solvers.evaluation import feature_vector, metrics


class Metric(enum.Enum):
    FID = "FID"
    MIND = "MIND"

    def __str__(self):
        return self.value

    def get_func(self):
        match self.value:
            case "FID": return metrics.frechet_inception_distance
            case "MIND": return metrics.monge_inception_distance


def gen_features(
        image_dir: str,
        batch_size: int = 64,
        device: str = "cuda",
        n_images: int = 0,
        output: str = None,
        **kwargs):
    print("Generating features...")

    if not output:
        output = image_dir + "/data/features.pkl"
    feature_vector.detect_image_features(
        image_dir,
        batch_size,
        device=device,
        n_images=n_images,
        save_path=output
    )
    print("Features generated")


def eval_features(
        sample_dir: str,
        ref_dir: str,
        metric: list[Metric],
        sample_output: str = None,
        ref_output: str = None,
        batch_size: int = 64,
        device: str = "cuda",
        n_images: int = 0,
        **kwargs
):
    print("Evaluating images...")
    if os.path.isdir(sample_dir):
        print("Detecting sample features...")
        x_hat = feature_vector.detect_image_features(
            sample_dir,
            batch_size,
            device=device,
            n_images=n_images,
            save_path=sample_output
        )
    else:
        print("Loading sample features...")
        x_hat = torch.load(sample_dir)

    if os.path.isdir(ref_dir):
        print("Detecting reference features...")
        x = feature_vector.detect_image_features(
            ref_dir,
            batch_size,
            detector=feature_vector.InceptionV3Detector(),
            device=device,
            n_images=n_images,
            save_path=ref_output
        )
    else:
        print("Loading reference features...")
        x = torch.load(ref_dir)

    print("Calculating metrics...")
    for m in metric:
        print(f"Calculating {m.name}")
        print(f"{m.name}: {m.get_func()(x, x_hat)}")

    print("Finished")


def main():
    parser = argparse.ArgumentParser(description="Computes a metric for given images")
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("-d", "--device", default="cuda", type=torch.device)
    parser.add_argument("-n", "--n_images", default=0, type=int)

    subparsers = parser.add_subparsers()

    gen_features_parser = subparsers.add_parser("gen-features")
    gen_features_parser.add_argument("image_dir", type=str)
    gen_features_parser.add_argument("-o", "--output", default=None, type=str)
    gen_features_parser.set_defaults(func=gen_features)

    eval_features_parser = subparsers.add_parser("eval-features")
    eval_features_parser.add_argument("sample_dir", type=str)
    eval_features_parser.add_argument("ref_dir", type=str)
    eval_features_parser.add_argument("-m", "--metric", action="append", default=[], type=Metric, choices=list(Metric))
    eval_features_parser.add_argument("-s", "--sample_output", default=None, type=str)
    eval_features_parser.add_argument("-r", "--ref_output", default=None, type=str)
    eval_features_parser.set_defaults(func=eval_features)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
