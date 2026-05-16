from abc import abstractmethod, ABC
import pickle

import torch
import tqdm

import dnnlib
import utils


class FeatureDetector(ABC):

    def __init__(self, n_features: int):
        self._n_features = n_features

    @abstractmethod
    def __call__(self, x: torch.Tensor):
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
        return self.model.to(x.device)(x, return_features=True)


def detect_image_features(
        image_dir: str,
        batch_size: int,
        detector: FeatureDetector,
        device: str | torch.device = "cpu",
        n_images: int = 0,
        save_path: str = None
):
    images = utils.ImageSampleDataset(image_dir, n_images)
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size)

    features = torch.zeros((len(images), detector.n_features))

    print("Calculating feature vectors...")
    for i, image_batch in tqdm.tqdm(enumerate(dataloader), total=len(images) // batch_size + 1):
        first_image_i = i * batch_size
        features[first_image_i: first_image_i + image_batch.shape[0]] = detector(image_batch.to(device))

    if save_path:
        torch.save(features, save_path)

    return features


# Test code
if __name__ == "__main__":
    features = detect_image_features(
        image_dir="../data/image_testing/em/50NFE/images",
        batch_size=48,
        detector=InceptionV3Detector(),
        device="cuda",
        save_path="./features.pkl",
        n_images=10000
    )

    print(utils.calculate_fid(features, features_2))
