from abc import abstractmethod, ABC
import pickle

import torch
import tqdm

from pi_solvers import dnnlib, utils
from pi_solvers.metrics import metrics


class FeatureDetector(ABC):

    def __init__(self, n_features: int):
        self._n_features = n_features

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass

    @abstractmethod
    def to(self, device: str | torch.device):
        pass

    @property
    def n_features(self):
        return self._n_features


# From EDM2 repository: https://github.com/NVlabs/edm2/tree/main
class InceptionV3Detector(FeatureDetector):

    def __init__(self, n_features: int = 2048):
        super().__init__(n_features)

        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

    def __call__(self, x: torch.Tensor):
        return self.model(x, return_features=True)

    def to(self, device: str | torch.device):
        self.model.to(device)


def detect_image_features(
        image_dir: str,
        batch_size: int,
        detector: FeatureDetector,
        device: str | torch.device = "cpu",
        n_images: int = 0,
        save_path: str = None
):
    images = utils.ImageSampleDataset(image_dir, n_images)
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=4, pin_memory=True)

    features = []
    detector.to(device)

    print("Calculating feature vectors...")
    for i, image_batch in tqdm.tqdm(enumerate(dataloader), total=len(images) // batch_size + 1):
        with torch.no_grad():
            feature_vectors = detector(image_batch.to(device)).to(torch.float64).to("cpu")
            features.append(feature_vectors)

    features = torch.cat(features)[:len(images)]

    if save_path:
        torch.save(features, save_path)

    return features


# Test code
if __name__ == "__main__":
    x_hat = detect_image_features(
        image_dir="../../refs/img64/train_64x64/train_64x64",
        batch_size=256,
        detector=InceptionV3Detector(),
        device="cuda",
        save_path="../../refs/img64/features.pkl",
    )
    x_hat = torch.load("../../data/image_testing/edm/50NFE_churn/data/features.pkl")
    x = torch.load("../../refs/img64/features.pkl")
    print(metrics.frechet_inception_distance(x, x_hat))
