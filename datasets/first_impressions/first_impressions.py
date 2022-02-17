"""first_impressions dataset."""

import dataclasses
from typing import Tuple
import cv2
import tensorflow_datasets as tfds
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

import skvideo 
skvideo.setFFmpegPath('/home/elyro2/personality-machine/ffmpeg-5.0-amd64-static')
import skvideo.io

# TODO(first_impressions): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(first_impressions): BibTeX citation
_CITATION = """
"""

LABELS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'interview']

@dataclasses.dataclass
class FirstImpressionsConfig(tfds.core.BuilderConfig):
  n_frames: int = 10
  img_size: Tuple[int, int] = (224, 398)
  # just rescale images to height
  # random cropping handled by NN

class FirstImpressions(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for first_impressions dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  # pytype: disable=wrong-keyword-args
  BUILDER_CONFIGS = [
      # `name` (and optionally `description`) are required for each config
      FirstImpressionsConfig(
        name='base', 
        description='just rescale images to height (random cropping handled by NN); clip non-720x180 images', 
        img_size=(224,398),
        n_frames=10,
      ),
  ]
  # pytype: enable=wrong-keyword-args
  MANUAL_DOWNLOAD_INSTRUCTIONS = "TODO"

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(first_impressions): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(*self.builder_config.img_size, 3)),
            # O,C,E,A,N,I
            'label': tfds.features.Tensor(shape=(6,), dtype=tf.float64),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _parse_labels(self, labels_path, csv_path):
    with tf.io.gfile.GFile(labels_path, "rb") as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
    pd.DataFrame(data).to_csv(csv_path)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(first_impressions): Downloads the data and defines the splits
    path = dl_manager.manual_dir

    # TODO(first_impressions): Returns the Dict[split names, Iterator[Key, Example]]
    self._parse_labels(path / 'train/annotation_training.pkl', path / 'train/labels.csv')
    self._parse_labels(path / 'test/annotation_test.pkl', path / 'test/labels.csv')
    self._parse_labels(path / 'val/annotation_validation.pkl', path / 'val/labels.csv')
    return {
        'train': self._generate_examples(path / 'train'),
        'test': self._generate_examples(path / 'test'),
        'val': self._generate_examples(path / 'val'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(first_impressions): Yields (key, example) tuples from the dataset
    df = pd.read_csv(path / 'labels.csv', index_col=0)
    for img_path in path.glob('*.mp4'):
      f = img_path.name
      idx = df.loc[f]
      videodata = skvideo.io.vread(str(path / f))
      if videodata.shape[1:3] != (720, 1280):
        continue

      n_frames = self.builder_config.n_frames
      n = videodata.shape[0]//n_frames
      
      frames = videodata[0:(n_frames*n):n]
      
      label = np.array([idx[i] for i in LABELS])

      for i, frame in enumerate(frames):
        yield f"{f}_{i}", {
            'image': cv2.resize(frame, self.builder_config.img_size[::-1]),
            'label': label,
        }
