# Lint as: python3
"""General utility functions."""
import os.path
from typing import Any, Callable, Tuple, Union
import cv2
import numpy as np
import random
import skimage
import scipy
from PIL import Image
import skimage.morphology
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from sklearn.feature_extraction.image import extract_patches_2d

AUTOTUNE = tf.data.experimental.AUTOTUNE
BIAS = 0.5



def preprocess_dataset(dataset: Any,
                       train: bool,
                       sigma,
                       transformation: Callable[..., Any],
                       batch_size: int = 1,
                       shuffle: Union[int, None] = None,
                       repeat: int = 1) -> Any:
  """Preprocess dataset.
  """
  # dataset = dataset.repeat(repeat)
  if train:
    dataset = dataset.map(train_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.cache()
    dataset = dataset.map(transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, filename: random_flip(x, filename),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, filename: random_rotate(x, filename),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, filename: add_noise(x, sigma, filename),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  else:
    dataset = dataset.map(test_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, filename: add_noise(x, sigma, filename),
                          num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


  return dataset

def random_rotate(image, filename):
  random_k = np.random.randint(0, 4)
  image = tf.image.rot90(
      image, k=random_k, name=None
    )

  return image, filename


def random_flip(image, filename):
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  return image, filename


def random_crop(image, patch_size, filename):
  return tf.image.random_crop(image, size=[patch_size, patch_size, 3]), filename


def train_parse_function(filename):
  # tf.print(filename)
  image_string = tf.io.read_file(filename)
  #Don't use tf.image.decode_image, or the output shape will be undefined
  image = tf.io.decode_jpeg(image_string, channels = 3)
  #This will convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)

  return image, filename

def test_parse_function(filename):
  image_string = tf.io.read_file(filename)
  #Don't use tf.image.decode_image, or the output shape will be undefined
  image = tf.image.decode_jpeg(image_string, channels = 3)
  #This will convert to float values in [0, 1]
  image = tf.image.convert_image_dtype(image, tf.float32)

  return image, filename


def random_patch_and_noise(lr, patch_size):
  """Crops a random patch from the image and adds Gaussian noise to it."""
  lr = lr
  sigma = tf.random.uniform([1],
                            minval=0,
                            maxval=2,
                            dtype=tf.dtypes.float32)
  channels = 3
  sigma = 35./255.
  lr = tf.image.random_crop(lr, size=[patch_size, patch_size, channels])
  noise = tf.random.normal(
      tf.shape(lr), mean=0, stddev=sigma, dtype=tf.float32)

  lr = tf.cast(lr, tf.float32)

  noisy = lr + noise
  return noisy, lr, sigma

def add_noise(lr, sigma, filename):
  """Adds Gaussian noise to the input image."""
  lr = lr
  # sigma = tf.random.uniform([1],
  #                           minval=0.5,
  #                           maxval=1,
  #                           dtype=tf.dtypes.float32)
  sigma = sigma/255.

  noise = tf.random.normal(
      tf.shape(lr), mean=0.0, stddev=sigma, dtype=tf.float32)

  lr = tf.cast(lr, tf.float32)
  # lr = lr - BIAS
  noisy = lr + noise
  # noisy = tf.clip_by_value(noisy, clip_value_min=0, clip_value_max=1)
  return noisy, lr, filename


def large_patch_noclip_dataset(
    patch_size: int = 160,
    batch_size: int = 1,
    sigma: float = 35.,
    shuffle=None,
    noise_sigma: Union[Tuple[int, int], float] = 1.0,
    cell=None) -> Tuple[Any, Any]:
  """Create dataset of patches for denoising.

  Args:
    patch_size: input dataset.
    batch_size: Integer, size of batch.
    shuffle: Integer, size of buffer for random shuffling.
    noise_sigma: Float, standard deviation of noise added.

  Returns:
    Denoising dataset, Tuple[train_dataset, validation_dataset]
  """

  Scale = False
  train_transform = lambda image, filename: random_crop(image, patch_size, filename)
  file_list = []
  bsr_path = "./data/bsr/"
  waterloo_path = "./data/waterloo_patches/"
  flickr_path = "./data/Flickr2K_HR_jpeg/"
  div2k_path = "./data/DIV2K_jpg/"

  for dirname, subdirs, filenames in os.walk(bsr_path):
    for filename in filenames:
      fullpath = os.path.join(dirname, filename)
      if ".jpg" in fullpath:
        file_list.append(fullpath)

  for dirname, subdirs, filenames in os.walk(waterloo_path):
    for filename in filenames:
      fullpath = os.path.join(dirname, filename)
      if ".jpg" in fullpath:
        file_list.append(fullpath)

  for dirname, subdirs, filenames in os.walk(flickr_path):
    for filename in filenames:
      fullpath = os.path.join(dirname, filename)
      if ".jpg" in fullpath:
        file_list.append(fullpath)
        
  for dirname, subdirs, filenames in os.walk(div2k_path):
    for filename in filenames:
      fullpath = os.path.join(dirname, filename)
      if ".jpg" in fullpath:
        file_list.append(fullpath)
  full_size = int(len(file_list)//batch_size)*batch_size
  file_list = file_list[:full_size]
  print(len(file_list))
  ds_train = tf.data.Dataset.from_tensor_slices(np.random.permutation(file_list))

  ds_train = preprocess_dataset(
      ds_train, True, sigma, train_transform, batch_size, shuffle=len(file_list), repeat=100)

  ds_val_cbsd = tf.data.Dataset.list_files("./data/CBSD68/*.png")
  ds_val_cbsd = preprocess_dataset(ds_val_cbsd, False, sigma, add_noise, 1, shuffle=None, repeat=10)

  ds_val_kodak = tf.data.Dataset.list_files("./data/Kodak24/*.png")
  ds_val_kodak = preprocess_dataset(ds_val_kodak, False, sigma, add_noise, 1, shuffle=None, repeat=10)

  ds_val_urban = tf.data.Dataset.list_files("./data/*.png")
  ds_val_urban = preprocess_dataset(ds_val_urban, False, sigma, add_noise, 1, shuffle=None, repeat=10)

  ds_val_mcmaster = tf.data.Dataset.list_files("./data/*.jpg")
  ds_val_mcmaster = preprocess_dataset(ds_val_mcmaster, False, sigma, add_noise, 1, shuffle=None, repeat=10)

  ds_val = [ds_val_cbsd, ds_val_kodak, ds_val_urban, ds_val_mcmaster]
  return ds_train, ds_val




