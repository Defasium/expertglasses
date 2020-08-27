#!/usr/bin/python3
'''Expert eyeglasses recommendation system class.

In this module the eyeglasses recommendation system was implemented.
The task of recommending eyeframes based on people's appearance is quite hard
and there are no similar projects available at the moment. This system takes
into account over 20 facial attributes, maps them into eyeglasses features
with the help of expert module (which consists with over 40 branches). Expert
module apply written beforehand rules to get necessary mappings. High
interpretability of such approach guaranties user's understanding and loyality
towards the system.

To get necessary attributes from the photo this system uses bunch of machine
learning algorithms and existing free services, i.e BetaFaceAPI and FACE++.
FacePlusPlus allows to locate bounding boxes of faces in images, while BetaFace
uses various classifiers to get most of the secondary-importance features. So
to use this model, you should have access to the internet.

To detect face shape probabilities, iris color, forehead size, jawtype type, skintone
the system uses own pretrained convolutional neural networks (CNNs). This models run
on your local machine on CPU.

To get all necessary eyeframes attributes, the large dataset (>8k records) of eyeframes
was parsed and processed. Because in real life there are not so many eyeframe models in
the local shop available, the generation of unique eyewear by given features was implemented.
The system use conditional GANs followed by Super Resolution GAN to create non-existing
high-definition images of the eyeframes.

At the current state expert recommendation system has already implemented:
    * Loading and updating the images with faces from file system or URLs
    * Extraction of facial attributes from the image
    * Explanation module, which helps to understand, how reccomendations are formed
    * Visualization of top-6 most corresponding eyeframes by shape, shape and color,
    random best-selling
    * Generating of unique eyeframes with Generative Adversarial Networks (GAN)
    * Caching previous results, so old photos will be processed immediately
    * English and russian localization for the interface

Examples:
    To use this system, you simply import class in your python code:
        # from expert_backend import ExpertEyeglassesRecommender

    After that you create an instance of this class with specified path to the image:
        # ins = ExpertEyeglassesRecommender('test.jpg')

    By passing a `lang` parameter you can specify language, which will be used at explanation step:
        # ins = ExpertEyeglassesRecommender('test.jpg', lang='en')

    Initialization of class may take quite a long time (from 20 second up to 2 minutes)
    After initialization your recomendations will be completed and you can get top 6
    best images of eyeglasses in the following way:
        # ins.plot_recommendations()

    To get explanation of the system try:
        # print(ins.description)

    To work with new image try:
        # ins.update_image('test2.jpg')
        # ins.expert_module()

    You can also pass an url:
        # ins.update_image(
                'https://github.com/Defasium/expertglasses/blob/master/assets/gui.png?raw=true')
        # ins.expert_module()

Todo:
    * Implementation of searching similar glasses
    * Implementation of initial preferences in style of eyeglasses
    * Implementation of simple eyeglasses try-on
    * Implementation of taking images from the web-camera
    * Train an eyebrow's shape classifier
    * Replace BetaFace service with own classifiers

.. _Expert eyeglasses recommendation system with Generative Adversarial Networks:
   https://github.com/Defasium/expertglasses

'''

import base64
import json
import os
from collections import Counter
from io import BytesIO

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from skimage import io

from faceplusplus import extract_features, rotate_image
from expert_and_explanation import translate_facevec2eyeglassesvec as map_face2glass
from model_architecture import build_network
from shufflenet_and_gans.common import resolve_single
from shufflenet_and_gans.srgan import generator

VERSION = __version__ = '0.2.5 Released 27-August-2020'

def change_hue(img, value):
    '''Changes saturation of an image multiple time of given value.

        Args:
            img (numpy.ndarray): image of shape (width, height, 3).
            value (float): saturation fraction (0. - black/white image,
                                                1. - unmodified,
                                                >1. - oversaturated image,
                                                0..1 - something in-between)
        Returns:
            image (numpy.ndarray): processed image.

    '''
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_value, s_value, v_value = cv2.split(imghsv)
    s_value *= value
    s_value = np.clip(s_value, 0.0, 1.0)
    imghsv = cv2.merge([h_value, s_value, v_value])
    return np.clip(cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR), 0.0, 1.0)


class ExpertEyeglassesRecommender:
    '''Class implementing expert eyeglasses recommendation system.

    The task of recommending eyeframes based on people's appearance is quite hard
    and there are no similar projects available at the moment. This system takes
    into account over 20 facial attributes, maps them into eyeglasses features
    with the help of expert module (which consists with over 40 branches). Expert
    module apply written beforehand rules to get necessary mappings. High
    interpretability of such approach guaranties user's understanding and loyality
    towards the system.

    Attributes:
        database (pandas.DataFrame): Database with eyeglasses, with more than 8k records.
        description (str): Explanation of created recommendations by given user.
        eyeglasses_shape_vector (numpy.ndarray): 31-dimensional feature vector, defining shape
        eyeglasses_color_vector (numpy.ndarray): 15-dimensional feature vector, defining color

    '''
    __slots__ = ['database', 'description', 'eyeglasses_shape_vector', 'eyeglasses_color_vector',
                 'error_occured', 'error_message', '_request_num', '_vectors', '_logger',
                 '_verbose', '_session', '_graph', '_prefix', '_face_vector', '_cache',
                 '_hash', '_tags', '_models', '_shapevec', '_colorvec', '_precompcv1', '_lang',
                 '_precompcv2', '_precomp', '_features', '_eyes', '_saturated_eyes', '_image',]

    def __init__(self, image, window=None, logger=None, verbose=False, lang='ru'):
        '''Constructor of recomendation class. In constructor you should always define path
        to the photo with face

        Args:
            image (str): path to the image file.
            window (PySimpleGUI.Window class, optional): Copy of window instance,
            which is necessary for progress bar updates in GUI realization.
            logger (logging.Logger class, optional): Logger instance for logging events.
            verbose (bool, optional): If True, prints info messages.
            lang (str, default='ru'): Language of recommendation's explanations. Supported
            values are ['en', 'ru'].
        '''
        self._request_num = 0
        self._vectors = dict()
        self._logger = logger
        self._verbose = verbose
        self._session = tf.Session()
        self._graph = tf.get_default_graph()
        self._prefix = os.path.dirname(os.path.abspath(__file__))
        self._face_vector = None
        self._lang = lang.strip().lower()
        self.eyeglasses_shape_vector = None
        self.eyeglasses_color_vector = None
        self.description = ''

        # if localization not supported, raise exception
        supported_localizations = [lang[:2] for lang in os.listdir(os.path.join(self._prefix,
                                                                                'lang'))]
        if self._lang not in supported_localizations:
            raise NotImplementedError('Language %s is not currently supported!' % self._lang)

        # if there are cached results from last execution, load them in memory
        if os.path.exists(os.path.join(self._prefix, 'utils/cached.gz')):
            self._cache = joblib.load(os.path.join(self._prefix, 'utils/cached.gz'))
            self._tags = {k: v[-1] for k, v in self._cache.items()}
            # for each hash there are corresponding facial attributes,
            # but no eyeglasses features, that's why we repeatedly call expert module
            for current_hash in self._cache.keys():
                self._hash = current_hash
                self.expert_module()
        else:
            self._cache = dict()  # hash table with cache
            self._tags = dict()

        # update progress bar in gui
        if window is not None:
            window['progbar'].update_bar(10)

        # image loading, api request to faceplusplus, image alignment
        self.update_image(image)
        if window is not None:
            window['progbar'].update_bar(20)

        # loading models into RAM, requiring ~400-600MB
        self._models = []
        models_path = os.path.join(self._prefix, 'models/')

        # for each pretrained model construct necessary architecture and load weights
        for model in sorted(os.listdir(models_path)):
            if self._logger is not None:
                self._logger.info('Loading model: %s', model)
            # TODO; change string methods to regex patterns
            # 3.h5 -> 3
            embed_size = int(model.split('_')[-1].split('.')[0])
            # 128x128x3 - > (128, 128, 3)
            input_shape = [int(s) for s in model.split('_')[-2].split('x')]
            # loading in RAM, so no OOM hapens
            with self._graph.as_default():
                with self._session.as_default():
                    with tf.device('/cpu:0'):
                        network = build_network(input_shape, embed_size)
                        network.load_weights(os.path.join(models_path, model))
            self._models.append(network)
            if window is not None:
                window['progbar'].update_bar(20 + 8 * len(self._models))

        # loading Conditional Generative Adversarial Network for eyeglasses generation
        if os.path.exists(os.path.join(self._prefix, 'utils/cgan.h5')):
            if self._logger is not None:
                self._logger.info('Loading model: conditional gan')
            with self._graph.as_default():
                with self._session.as_default():
                    with tf.device('/cpu:0'):
                        # conditional gan
                        cgan = tf.keras.models.load_model(os.path.join(self._prefix,
                                                                       'utils/cgan.h5'))
                        self._models.append(cgan)
            if window is not None:
                window['progbar'].update_bar(20 + 8 * len(self._models))
        else:
            self._models.append(None)

        # loading Super Resolution Generative Adversarial Network for upscaling
        if os.path.exists(os.path.join(self._prefix, 'utils/srgan.h5')):
            if self._logger is not None:
                self._logger.info('Loading model: super resolution gan')
            with self._graph.as_default():
                with self._session.as_default():
                    with tf.device('/cpu:0'):
                        upscaler_gan = generator()
                        upscaler_gan.load_weights(os.path.join(self._prefix, 'utils/srgan.h5'))
            self._models.append(upscaler_gan)  # super resolution gan
            if window is not None:
                window['progbar'].update_bar(75)
        else:
            self._models.append(None)

        # api request to betafaceapi, facial attributes construction and mapping to eyeglasses ones
        self.expert_module()
        if window is not None:
            window['progbar'].update_bar(90)

        # loading eyeframes with attributes
        self.database = pd.read_csv(os.path.join(self._prefix, 'data/database.csv.gz'),
                                    index_col=None)
        self._shapevec = pd.read_csv(os.path.join(self._prefix, 'data/shape_vectors.csv.gz'),
                                     index_col=None)
        self._colorvec = pd.read_csv(os.path.join(self._prefix, 'data/color_vectors.csv.gz'),
                                     index_col=None)
        if window is not None:
            window['progbar'].update_bar(93)

        # normalizing eyeglasses vectors of real images
        self._precompcv1 = self._shapevec[self._shapevec.index30][self._shapevec.columns[1:-1]]\
            .apply(lambda x: x.astype(float) / np.linalg.norm(x.astype(float)), axis=1).values
        self._precompcv2 = self._shapevec[~self._shapevec.index30][self._shapevec.columns[1:-1]]\
            .apply(lambda x: x.astype(float) / np.linalg.norm(x.astype(float)), axis=1).values
        self._precomp = self._colorvec[self._colorvec.columns[1:]] \
            .apply(lambda x: x.astype(float) / np.linalg.norm(x.astype(float)), axis=1).values
        if window is not None:
            window['progbar'].update_bar(99)

    def save(self):
        '''Saving cached results to the disk'''
        joblib.dump(self._cache, os.path.join(self._prefix, 'utils/cached.gz'), 3, 4)

    def __get_vecs(self) -> (np.ndarray, np.ndarray):
        '''Get eyeglasses vectors for the current image file. In case there are no
        calculated vectors, make a call to an expert module

        '''
        try:
            return self.eyeglasses_shape_vector.copy(), self.eyeglasses_color_vector.copy()
        except:
            self.expert_module()
            return self.eyeglasses_shape_vector.copy(), self.eyeglasses_color_vector.copy()

    def update_facevector(self, facevector):
        '''Updates facial attributes from photo. Also updates attributes in cache.

                Args:
                    facevector (dict with str as keys): Dictionary with 22 facial attributes.

                Returns:
                    None.

        '''

        # delete from cache, so when we call expert module, it won't return old vectors
        try:
            del self._vectors[self._hash]
            tmp = self._cache[self._hash]
            self._cache[self._hash] = facevector, *tmp[1:]
        except KeyError:
            pass
        self._face_vector = facevector
        self.expert_module()

    def generate_unique(self, show=True, block=True):
        '''Generates unique eyeframes with GANs.

                Args:
                    show (bool): When True shows matplotlib's plot with generated image.
                    block (bool, optional): Block further command execution by matplotlib or not.
                        The default value is True
                Returns:
                    image (np.ndarray): Generated image with shape (256, 256, 3).

        '''

        # sanity check for existing models
        if self._models[-1] is None or self._models[-2] is None:
            if self._verbose:
                print('No GANs found! You need gan models to generate unique images')
            if self._logger is not None:
                self._logger.error('No GANs found! You need gan models to generate unique images')
            return np.zeros_like((256, 256, 3))

        shapevec, colorvec = self.__get_vecs()

        # normalize vectors, so our gans will generate adequate results
        slices_sv = [(0, 10), (10, 13), (13, 16), (16, 19), (23, 28), (28, 30)]
        slices_cv = [(0, 14)]
        for slice_left, slice_right in slices_sv:
            argmax = shapevec[slice_left:slice_right].argmax()
            shapevec[slice_left:slice_right] = 0.0
            shapevec[argmax] = 1.0
        for slice_left, slice_right in slices_cv:
            argmax = colorvec[slice_left:slice_right].argmax()
            colorvec[slice_left:slice_right] = 0.0
            colorvec[argmax] = 1.0
        shapevec = np.clip(shapevec, -0.1, 1.0).reshape(1, -1)
        colorvec = np.clip(colorvec, -0.1, 1.0).reshape(1, -1)

        # generate image from noise, models[-1] is SRGAN, while models[-2] is CGAN
        with self._graph.as_default():
            with self._session.as_default() as sess:
                image = sess.run(
                    resolve_single(self._models[-1],
                                   (-self._models[-2] \
                                    .predict([np.random.randn(1, 50),
                                              np.concatenate([shapevec, colorvec], axis=1)]
                                             )[0]) * 127.5 + 127.5))

        # assign flipped second half of image to first half to utilize asymmetry
        image[:, :127] = image[:, -1:128:-1]

        # plot images with blocking or not
        if show:
            plt.imshow(image)
            plt.show(block=block)
        return image

    def distances(self, strategy='standart'):
        '''Calculates cosine similarities between constructed eyeglasses features and features
        from database.

                Args:
                    strategy (str, optional): The possible values are 'standart', 'factorized',
                    'factorized_plus', 'color_only' and 'shape_only'.
                    The default value is 'factorized'.

                Returns:
                    Vector of distances with size of dataset and corresponing indexes in it.

        '''

        shapevec, colorvec = self.__get_vecs()

        # filter by gender
        if shapevec[-1] > 0:
            sub_df = self._shapevec[self._shapevec.index30].copy()
            idx = sub_df.index.values
            sub_df = self._precompcv1
        else:
            sub_df = self._shapevec[~self._shapevec.index30].copy()
            idx = sub_df.index.values
            sub_df = self._precompcv2

        # in standart strategy feature with the biggest value impacts the results the most
        if strategy == 'standart':
            shape_dist = sub_df.dot(shapevec[:-1] / np.linalg.norm(shapevec[:-1]))
            color_dist = self._precomp[idx].dot(colorvec / np.linalg.norm(colorvec))
            return shape_dist * color_dist, idx

        # in factorized strategy features are divided into three main groups:
        # shape of eyewear, its rim and other features, thus making results more various
        if strategy == 'factorized':
            slices = [(0, 10), (10, 13), (13, 30)]
            final_dist = self._precomp[idx].dot(colorvec / np.linalg.norm(colorvec))
            for slice_left, slice_right in slices:
                final_dist *= sub_df[:, slice_left:slice_right] \
                    .dot(shapevec[slice_left:slice_right] /
                         np.linalg.norm(shapevec[slice_left:slice_right]))
            return final_dist, idx

        # in factorized plus strategy every feature greatly influence the finite result
        # thus providing many varios nonsimilar eyeframes
        if strategy == 'factorized_plus':
            slices = [(0, 10), (10, 13), (13, 19), (19, 20), (20, 21),
                      (21, 22), (22, 23), (23, 28), (28, 30)]
            final_dist = self._precomp[idx].dot(colorvec / np.linalg.norm(colorvec))
            for slice_left, slice_right in slices:
                final_dist *= sub_df[:, slice_left:slice_right]\
                                .dot(shapevec[slice_left:slice_right] /
                                     np.linalg.norm(shapevec[slice_left:slice_right]))
            return final_dist, idx

        # in color only strategy the resulting images will have the same color as in color vector
        if strategy == 'color_only':
            color_dist = self._precomp[idx].dot(colorvec / np.linalg.norm(colorvec))
            return color_dist, idx

        # shape only strategy is similar to standart but doesn't take into account the color
        if strategy == 'shape_only':
            shape_dist = sub_df.dot(shapevec[:-1] / np.linalg.norm(shapevec[:-1]))
            return shape_dist, idx

    def plot_recommendations(self, strategy='factorized', block=True, return_links=False):
        '''Plot top 6 eyeframes recommendations from database by given strategy.

                Args:
                    strategy (str, optional): The possible values are 'standart', 'factorized',
                    'factorized_plus', 'color_only', 'shape_only' and 'most_popular'.
                    The default value is 'factorized'.
                    block (bool, optional): Block further command execution by matplotlib or not.
                    The default value is True
                    return_links (bool, optional): If True then return links to recommended images.
                    The default value is False

                Returns:
                    None.

        '''

        shapevec, _ = self.__get_vecs()

        # for ab-testing we compare recommendations with the ones randomly chosen best selling ones
        if strategy == 'most_popular':
            if shapevec[-1] > 0:
                directory = 'abtest/man'
            else:
                directory = 'abtest/woman'
            ims = np.random.choice([os.path.join(directory, f) for f in os.listdir(directory)],
                                   6, replace=False)
        else:
            dist, idx = self.distances(strategy)
            # sort by index and take 6 biggest values, i.e. the most similar
            top6 = idx[dist.argsort()[-6:][::-1]]
            ims = self.database.iloc[top6].image_link.tolist()

        # lambda function for converting image links to the necessary format
        pretty = lambda x: 'http:' + x if x[0] == '/' else x

        # if interested in image-links only, simply return them
        if return_links:
            return [pretty(img) for img in ims]

        fig = plt.figure(figsize=(21, 14))
        axes = fig.subplots(2, 3, sharex='col', sharey='row')

        # two rows
        for i in range(2):
            # three columns
            for j in range(3):
                axes[i, j].text(200, 300, str(i * 3 + j + 1),
                                fontsize=18, ha='center')

                # download from internet
                img = io.imread(pretty(ims[i * 3 + j]))

                # guaranty that all images have the same ratio, if not, pad it with white color
                if img.shape[1] / img.shape[0] != 1.5:
                    offset = int((2/3 - img.shape[0] / img.shape[1]) * img.shape[1] // 2)
                    img = cv2.copyMakeBorder(img, offset, offset, 0, 0,
                                             cv2.BORDER_CONSTANT,
                                             value=(255, 255, 255))
                axes[i, j].imshow(cv2.resize(img, (375, 250)))
        plt.show(block=block)

    def update_image(self, image):
        '''Update current image in the system by given image path.

                Args:
                    image (str of file obj): A filename or URL (string), pathlib.
                    Path object or a file object. The file object must implement
                    read(), seek(), and tell()methods, and be opened in binary mode.

                Returns:
                    None.

        '''

        if self._verbose:
            print('[INFO] Updating new image...')
        if self._logger is not None:
            self._logger.info('Updating new image...')

        if isinstance(image, str):
            self._image = io.imread(image)[:, :, ::-1]
        else:
            self._image = np.array(Image.open(image))[:, :, ::-1]

        # hash is the string represantation of the downscaled image
        self._hash = str(list(cv2.resize(self._image, (32, 32))))

        # if the system has this image in cache already, api request is not necessary
        if self._hash in self._cache:
            if self._logger is not None:
                self._logger.info('Image has already fetched...')
            self._face_vector, self._features, self._image, \
            self._tags[self._hash] = self._cache[self._hash]
            return

        # rest-api to get points and other stuff
        if self._verbose:
            print('[INFO] extracting features...')
        if self._logger is not None:
            self._logger.info('Extracting features...')
        self._features = extract_features(image)
        rect = self._features[-1]['faces'][0]['face_rectangle']

        # get uppermost forehead point
        upper_forehead_coord = self._features[3]['face_hairline_72']

        # get cropped eyes
        if self._verbose:
            print('[INFO] cropping eyes...')
        if self._logger is not None:
            self._logger.info('Cropping eyes...')
        landmarks = self._features[-1]['faces'][0]['landmark']
        self._eyes = []
        for k in ['left', 'right']:
            center = landmarks['%s_eye_center' % k]['y']
            left, right = landmarks['%s_eyebrow_left_corner' % k]['x'], \
                          landmarks['%s_eyebrow_right_corner' % k]['x']
            offset = (right - left) // 2
            self._eyes.append(self._image[center - offset:center + offset + (right - left) % 2,
                                          left:right].copy())
        self._eyes = np.array([cv2.resize(eye, (64, 64), cv2.INTER_LINEAR) for eye in self._eyes]
                              ) / 255.
        self._saturated_eyes = np.array([change_hue(eye.astype(np.float32), 1.5)
                                         for eye in self._eyes])

        # old face box - rect - crops forehead area, fixs it
        if self._verbose:
            print('[INFO] cropping face...')
        if self._logger is not None:
            self._logger.info('Cropping face...')
        height = rect['top'] + rect['height'] * 11 // 10 - \
                 upper_forehead_coord['y'] + int(self._image.shape[0] / 8)
        diff = (height - rect['width']) // 2

        # if diffenece is negative then it is necessary to add borders
        diff_y = upper_forehead_coord['y'] - int(self._image.shape[0] / 8)
        diff_y2 = rect['top'] + rect['height'] * 11 // 10
        if diff_y < 0:
            self._image = cv2.copyMakeBorder(self._image, -diff_y, 0, 0, 0, cv2.BORDER_CONSTANT)
            diff_y = 0
        if diff_y2 > self._image.shape[0]:
            self._image = cv2.copyMakeBorder(self._image, 0, diff_y2 - self._image.shape[0],
                                             0, 0, cv2.BORDER_CONSTANT)
        if rect['left'] - diff > 0:
            self._image = self._image[diff_y:diff_y2,
                                      rect['left'] - diff:rect['left'] + rect['width'] + diff]
        else:
            self._image = self._image[diff_y:diff_y2, :rect['left'] + rect['width'] + diff]
            self._image = cv2.copyMakeBorder(self._image, 0, 0,
                                             diff - rect['left'], 0, cv2.BORDER_CONSTANT)

        # rotate face
        self._image = rotate_image(self._image,
                                   self._features[-1]['faces']
                                   [0]['attributes']['headpose']['roll_angle'])

    def get_attributes(self):
        '''Get facial attributes by api request to betafaceapi.
        Only 500 images can be requested daily from free api, which is used here.
        Request usually takes from 3 to 5 seconds.
        '''

        # if already have tags from rest api then just simply return them
        if self._hash in self._tags:
            return self._tags[self._hash]

        # otherwise get another request
        self._request_num += 1

        is_success, buffer = cv2.imencode('.jpg', self._image)
        if self._logger is not None:
            self._logger.debug('Image convertion code: %d', is_success)
        b_io = BytesIO(buffer)
        b_io.seek(0)
        b64string = base64.b64encode(b_io.read())

        # using free api key
        params = {
            'api_key': 'd45fd466-51e2-4701-8da8-04351c872236',
            'file_base64': str(b64string)[2:-1],
            'detection_flags': 'classifiers',
        }

        resp = requests.post('https://www.betafaceapi.com/api/v2/media',
                             data=json.dumps(params), headers={
                                 'accept': r'application/json',
                                 'Content-Type': r'application/json'})
        try:
            answer = json.loads(resp.text)
        except json.JSONDecodeError as exception:
            if self._logger is not None:
                self._logger.error('Error at API request to betaface: %s', exception)
        try:
            self._tags[self._hash] = {d['name']: d['value'] for d in
                                      answer['media']['faces'][0]['tags']}
            return self._tags[self._hash]
        except KeyError as exception:
            if self._logger is not None:
                self._logger.error('Error at API request to betaface: Not face found or '
                                   'check dailyrequests to rest-api: %d/500\n %s',
                                   self._request_num, exception)
        return None

    def get_facevector(self):
        '''Construct dictionary of facial attributes. At this step, parsed attributes
        from betafaceapi are concatenated with ones, obtained
        from the pretrained neural networks.

                Returns:
                    Dictionary with facial attributes.

        '''

        if self._verbose:
            print('[INFO] getting face vector...')
        if self._logger is not None:
            self._logger.info('Getting face vector...')

        # if face vector has already been in cache, simply return it
        if self._hash in self._cache:
            self._face_vector = self._cache[self._hash][0]
            return self._face_vector

        # otherwise construct it from start
        self._face_vector = dict(
            faceshape=self.__get_faceshape(),
            ratio=self.__get_faceratio(),
            jawtype=self.__get_jawtype(),
            beard=self.__get_beard(),
            doublechin=self.__get_doublechin(),
            highcheeckbones=self.__get_hch(),
            chubby=self.__get_chubby(),
            eyebrows_thickness=self.__get_ebrthickness(),
            eyebrows_shape=self.__get_ebrshape(),
            nose_size=self.__get_nose(),
            eyes_narrow=self.__get_eyes_narrow(),
            eyes_iris=self.__get_eyes_iris(),
            forehead=self.__get_forehead(),
            bangs=self.__get_bangs(),
            lips=self.__get_lips(),
            mustache=self.__get_mustache(),
            bald=self.__get_bald(),
            hair=self.__get_hair(),
            skintone=self.__get_skintone(),
            race=self.__get_race(),
            paleskin=self.__get_pale(),
            gender=self.__get_gender()
        )
        self._cache[self._hash] = self._face_vector, self._features, \
                                  self._image, self._tags[self._hash]
        return self._face_vector

    def __get_base_vectors(self):
        self.eyeglasses_shape_vector = np.array([
            .1,  # index 0 shape == rectangle
            .1,  # index 1 shape == square
            .1,  # index 2 shape == oval
            .1,  # index 3 shape == round
            .1,  # index 4 shape == aviator
            .1,  # index 5 shape == cat eye
            .1,  # index 6 shape == other
            .1,  # index 7 shape == clubmaster
            .1,  # index 8 shape == walnut
            .1,  # index 9 shape == wayfarer
            1.,  # index 10 rim == fullrim
            .5,  # index 11 rim == semirim
            .1,  # index 12 rim == rimless
            .1,  # index 13 height == small
            .1,  # index 14 height == medium
            .1,  # index 15 height == large
            .1,  # index 16 width == small
            .1,  # index 17 width == medium
            .1,  # index 18 width == large
            .1,  # index 19 nosepads == yes
            .1,  # index 20 top_heavy == yes
            .1,  # index 21 angular == yes
            .5,  # index 22 thick == yes
            .1,  # index 23 uppershape == flat
            .1,  # index 24 uppershape == round
            .1,  # index 25 uppershape == smoothed
            .1,  # index 26 uppershape == roof
            .1,  # index 27 uppershape == angry
            .1,  # index 28 material == plastic
            .1,  # index 29 material == metal
            .0,  # index 30 male == yes
        ], dtype=np.float16)

        self.eyeglasses_color_vector = np.array([
            1.,  # index 0 color == black
            .1,  # index 1 color == brown
            .1,  # index 2 color == gold
            .1,  # index 3 color == blue
            .1,  # index 4 color == silver
            .1,  # index 5 color == red
            .1,  # index 6 color == green
            .1,  # index 7 color == purple
            .1,  # index 8 color == grey
            .1,  # index 9 color == transparent
            .1,  # index 10 color == pink
            .1,  # index 11 color == shiny
            .1,  # index 12 color == white
            .1,  # index 13 color == multicolor
            .1,  # index 14 color == colors_differ == yes
        ], dtype=np.float16)

        return self.eyeglasses_shape_vector, self.eyeglasses_color_vector

    def expert_module(self):
        '''Class method for translating facial features to eyeglasses features
        by consequentially applying expert rules.

                Returns:
                    tuple (numpy.ndarray, numpy.ndarray): sizes 31 and 15,
                    for eyeglasses shape and color features respectively.

        '''

        if self._hash in self._vectors:
            self.eyeglasses_shape_vector, \
            self.eyeglasses_color_vector, \
            self.description = self._vectors[self._hash]
            return self.eyeglasses_shape_vector, self.eyeglasses_color_vector
        s_vector, c_vector = self.__get_base_vectors()
        facevector = self.get_facevector()

        # translating facevector to eyeglasses vectors
        s_vector, c_vector, self.description = map_face2glass(facevector, s_vector,
                                                              c_vector, self._lang)
        self.eyeglasses_shape_vector, self.eyeglasses_color_vector = s_vector, c_vector
        self._vectors[self._hash] = s_vector, c_vector, self.description
        return s_vector, c_vector

    def __clone_images(self, color=False):
        # face on original image is too small, scaling face
        correction = int(120 / 545 * self._image.shape[0])
        faces = []

        # use different scaling factors to increase accuracy of predictions
        for factor in [0.5, 0.75, 1.25, 1]:
            cur_correction = int(correction * factor)
            faces.append(cv2.resize(self._image[cur_correction:,
                                                cur_correction // 2:-cur_correction // 2],
                                    (128, 128),
                                    cv2.INTER_LINEAR_EXACT).astype(np.float32) / 255.)

        # use grayscale images for prediction as well as every channel in images
        bface = []
        for face in faces:
            if color:
                face = np.vstack([face.reshape(1, 128, 128, 3)] + \
                                 [rotate_image(face, np.random.uniform(-5, 5)
                                               ).reshape(1, 128, 128, 3)
                                  for i in range(3)])
            else:
                face = np.vstack([cv2.cvtColor(face, cv2.COLOR_BGR2GRAY
                                               ).reshape(1, 128, 128, 1)] + \
                                 [face[:, :, i].reshape(1, 128, 128, 1) for i in range(3)])

            # also adding horizontally flipped faces
            if len(bface) > 1:
                bface = np.vstack([bface, face, face[:, :, ::-1, :]])
            else:
                bface = np.vstack([face, face[:, :, ::-1, :]])

        return bface

    def __get_faceshape(self):
        bface = self.__clone_images()

        # get 7-dimensional embedding for each face
        with self._graph.as_default():
            with self._session.as_default():
                embed = self._models[1].predict(bface)

        # clear some memory
        del bface

        # load models
        (bgmm, knn_minority), \
        (bgmj, knn_majority), \
        (bgmj2, knn_majority2), \
        svm, mapper = joblib.load('utils/mixture_plus_svm')

        # divide by 3 to control the level of how much we believe the model
        cnt = Counter(
            {k: v // 3 for k, v in zip(mapper, list(np.sum(svm.predict_proba(embed),
                                                           axis=0).astype(np.uint8)))})

        # compute differences in likelihoods of belonging embed to each gaussian mixture
        sub = np.subtract(bgmj.score_samples(embed), bgmm.score_samples(embed))
        sub2 = np.subtract(bgmj2.score_samples(embed), bgmm.score_samples(embed))
        sub3 = np.subtract(bgmj2.score_samples(embed), bgmj.score_samples(embed))

        # predict classes for each group and summarize
        try:
            cnt += Counter(knn_minority.predict(embed[np.where((sub < 0) & (sub2 < 0))]))
        except ValueError:
            pass
        try:
            cnt += Counter(knn_majority.predict(embed[np.where((sub3 < 0) & (sub > 0))]))
        except ValueError:
            pass
        try:
            cnt += Counter(knn_majority2.predict(embed[np.where((sub3 > 0) & (sub2 > 0))]))
        except ValueError:
            pass
        norm_coeff = sum(cnt.values())

        # return normed distribution over face shapes
        # first value is fraction, while second is shape
        return sorted([(v / norm_coeff, k) for k, v in cnt.items()], reverse=True)

    def __get_faceratio(self):
        ratio = self._features[1]['l_center'] / self._features[1]['w_center']
        if ratio > 1.36:
            return 'longer'
        if ratio >= 1.18:
            return 'normal'
        return 'wider'

    def __get_jawtype(self):
        # augment images to get more accurate results
        bface = self.__clone_images()

        # get embeddings for augmented images from neural network
        with self._graph.as_default():
            with self._session.as_default():
                embed = self._models[3].predict(bface)

        # load gaussian mixture model and mapping dictionary
        gmm, mapping = joblib.load(os.path.join(self._prefix, 'utils/jawgmm'))
        return mapping[np.argmax(gmm.predict_proba(embed).sum(axis=0))]

    def __get_beard(self):
        tags = self.get_attributes()
        return tags['beard']

    def __get_doublechin(self):
        tags = self.get_attributes()
        return tags['double chin']

    def __get_hch(self):
        tags = self.get_attributes()
        return tags['high cheekbones']

    def __get_chubby(self):
        tags = self.get_attributes()
        return tags['chubby']

    def __get_ebrthickness(self):
        points = self._features[-1]['faces'][0]['landmark']
        length = self._features[1]['l_center'] * self._features[1]['w_center']

        # thickness of eyebrows is average difference between points
        right_eyebrow_ly, right_eyebrow_uy = points['right_eyebrow_lower_middle'], \
                                             points['right_eyebrow_upper_middle']
        left_eyebrow_ly, left_eyebrow_uy = points['left_eyebrow_lower_middle'], \
                                           points['left_eyebrow_upper_middle']

        # calculate and normalize
        thickness = ((right_eyebrow_ly['y'] - right_eyebrow_uy['y'] +
                      left_eyebrow_ly['y'] - left_eyebrow_uy['y'])
                     / (2 * length) - 0.0025) / 5.3e-5
        if thickness > 1.0:
            return 'thick'
        if thickness > -3.0:
            return 'normal'
        return 'thin'

    def __get_ebrshape(self):
        # TODO; implement eyebrow shape detector
        tags = self.get_attributes()
        if tags['arched eyebrows'] == 'yes':
            return 'curly'
        return 'flat'

    def __get_nose(self):
        tags = self.get_attributes()
        if tags['big nose'] == 'yes':
            return 'big'
        if tags['pointy nose'] == 'yes':
            return 'long'
        return 'small'

    def __get_eyes_narrow(self):
        tags = self.get_attributes()
        return tags['narrow eyes']

    def __get_eyes_iris(self):
        eyes_model = self._models[0]

        # augment images to get more accurate results
        eyes = np.concatenate([self._eyes, self._saturated_eyes])
        beyes = []
        for eye in eyes:
            for _ in range(5):
                tmp = eye[np.random.randint(0, 16):-np.random.randint(1, 16),
                          np.random.randint(0, 16):-np.random.randint(1, 16)].copy()
                beyes.append(rotate_image(cv2.resize(tmp, (64, 64)),
                                          np.random.uniform(-45, 45)))
        beyes = np.array(beyes)

        # get embeddings for augmented and horizontally flipped images from neural network
        with self._graph.as_default():
            with self._session.as_default():
                embed = eyes_model.predict(np.concatenate([beyes, beyes[:, :, ::-1]]))

        # load gaussian mixture model and mapping dictionary
        gmm, mapping = joblib.load(os.path.join(self._prefix, 'utils/eyesgmm'))
        return mapping[np.argmax(gmm.predict_proba(embed).sum(axis=0))]

    def __get_forehead(self):
        # augment images to get more accurate results
        bface = self.__clone_images()

        # get embeddings for augmented images from neural network
        with self._graph.as_default():
            with self._session.as_default():
                embed = self._models[2].predict(bface)

        # load gaussian mixture model and mapping dictionary
        gmm, mapping = joblib.load(os.path.join(self._prefix, 'utils/fheadsgmm'))
        return mapping[np.argmax(gmm.predict_proba(embed).sum(axis=0))]

    def __get_bangs(self):
        tags = self.get_attributes()
        return tags['bangs']

    def __get_lips(self):
        tags = self.get_attributes()
        return 'big' if tags['big lips'] == 'yes' else 'normal'

    def __get_mustache(self):
        tags = self.get_attributes()
        return tags['mustache']

    def __get_bald(self):
        tags = self.get_attributes()
        return tags['bald']

    def __get_hair(self):
        tags = self.get_attributes()
        if tags['black hair'] == 'yes':
            return 'black'
        if tags['blond hair'] == 'yes':
            return 'blonde'
        if tags['brown hair'] == 'yes':
            return 'brown'
        if tags['gray hair'] == 'yes':
            return 'gray'
        return 'red'

    def __get_skintone(self):
        # augment images to get more accurate results
        bface = self.__clone_images(color=True)

        # get embeddings for augmented images from neural network
        with self._graph.as_default():
            with self._session.as_default():
                embed = self._models[4].predict(bface)

        # load gaussian mixture model and mapping dictionary
        gmm, mapping = joblib.load(os.path.join(self._prefix, 'utils/skintonegmm'))
        return mapping[np.argmax(gmm.predict_proba(embed).sum(axis=0))]

    def __get_race(self):
        tags = self.get_attributes()
        return tags['race']

    def __get_pale(self):
        tags = self.get_attributes()
        return tags['pale skin']

    def __get_gender(self):
        tags = self.get_attributes()
        return tags['gender']
