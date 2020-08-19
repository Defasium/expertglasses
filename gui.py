#!/usr/bin/python3
'''GUI for Expert eyeglasses recommendation system.

This module is a graphical user interface (GUI) in russian for the eyeglasses
recommendation system. It is powered by PySimpleGUI, which allows fast and simple
prototyping for cross-platform systems. The main purpose of this module is to
demonstrate features, such as:
    * Loading and updating the images with faces
    * Visualization of loaded images
    * Extraction of facial attributes from the image
    * Editing of most facial attributes to correct errors of classifiers
    * The description of expert module
    * Visualization of top-6 most corresponding eyeframes
    * Generating of unique eyeframes with Generative Adversarial Networks (GAN)
    * Caching previous results, so old photos will be processed immediately

Example:
    To launch script, you should use the following command from the terminal:
        $ python gui.py
    In the first popup window you should load an image to process, i.e. the fast start,
    module would automatically calculate all the necessary stuff in background. After
    that all machine learning models will be loaded in RAM (loading usually takes from 30
    seconds up to 2 minutes). When progressbar will be filled, the main interface will appear.

Todo:
    * Add english version of the GUI, which will be available as option at the starting screen
    * Implementation of search of similar glasses, by clicking on eyeframes' image
    * Implementation of toggle bars controls for the extracted face shape
    * Implementation of summarizing results of explanation module
    * Implementation of initial preferences in style of eyeglasses
    * Implementation of simple eyeglasses try-on
    * Implementation of taking images from the web-camera
    * Implementation of uploading photos from the internet by url

.. _Expert eyeglasses recommendation system with Generative Adversarial Networks:
   https://github.com/Defasium/expertglasses

'''
import base64
import logging
import sys
from functools import partial
from io import BytesIO

import PySimpleGUI as sg
import cv2

from expert_backend import ExpertEyeglassesRecommender

VERSION = __version__ = '0.2.0 Released 26-May-2020'

# font-styles
FONT_STYLE = 'Helvetica'
BIG_FONT = (FONT_STYLE, 18)
MEDIUM_FONT = (FONT_STYLE, 15)
SMALL_FONT = (FONT_STYLE, 12)
TINY_FONT = (FONT_STYLE, 10)

if __name__ == '__main__':
    # partial functions for some frequent elements to reduce repetitions in arguments
    text_element_left = partial(sg.Text, font=MEDIUM_FONT, justification='left')
    text_element_right = partial(sg.Text, font=MEDIUM_FONT, justification='right')
    button_element = partial(sg.Button, visible=True, enable_events=True)
    combo_element = partial(sg.Combo, font=MEDIUM_FONT, auto_size_text=True)

    # gets or creates a logger
    logger = logging.getLogger(__name__)

    # define file handler and set formatter
    file_handler = logging.FileHandler('logfile.log')

    # add file handler to logger
    logger.addHandler(file_handler)

    # layout for loading components Window
    layout0 = [[sg.Text('Загрузка модулей')],
               [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progbar')]]

    # layout for start screen
    layout = [[sg.Text('Выберите фотографию с лицом')],
              [sg.Input(key='file', visible=False, enable_events=True), sg.FileBrowse('Открыть')]]

    # create the Window
    win1 = sg.Window('Открытие файла', layout)
    _, values = win1.read()
    # need to destroy the window as it's still open
    win1.close()

    image_path = values['file']
    # check when no image was chosen
    if not image_path:
        logger.error('Вы не выбрали изображение! Завершение работы...')
        sys.exit()

    # window for loading of all necessary components
    window = sg.Window('Загрузка модулей', layout0)
    logger.info('You chose: %s', image_path)

    # initializing window and progressbar
    event, values = window.read(timeout=10)
    window['progbar'].update_bar(0)

    # create instance of ExpertEyeglassesRecommender with window callback to update progressbar
    ins = ExpertEyeglassesRecommender(image_path, window, logger)
    window['progbar'].update_bar(100)

    # need to destroy the window as it's still open
    window.close()

    # convert image to base64-encoded string, it is necessary to visualize in the main interface
    is_success, buffer = cv2.imencode('.png', cv2.resize(ins._image, (256, 256)))
    logger.debug('Image convertion code: %d', is_success)
    b_io = BytesIO(buffer)
    b_io.seek(0)
    b64string = base64.b64encode(b_io.read())

    # layout for main window
    layout2 = [[sg.Text('Экспертная рекомендация оправ', font=BIG_FONT)],
               [sg.Input(key='file', visible=False, enable_events=True),
                sg.FileBrowse('Обновить изображение'),
                button_element('Выделение признаков лица')],
               [button_element('Составление признаков оправы'),
                button_element('Вывод описания системы')],
               [sg.Image(data=b64string, size=(256, 256), pad=(64, 64), key='face_image')],
               [button_element('Сохранить'),
                button_element('Подбор оправ по базе'),
                button_element('Сгенерировать индивидуальную оправу')]]

    # creating main interface
    window = sg.Window('Экспертная рекомендация оправ',
                       layout2, resizable=True,
                       grab_anywhere=True, font=SMALL_FONT)

    # graphical interface work in synchronous mode so we need to wait
    # for each event or command in infinite loop
    while True:
        event, values = window.read()
        # updating image scenario
        if event == 'file':
            try:
                image_path = values['file']
                # checking if no file was chosen
                if image_path is not None and image_path != '':
                    logger.info('Image to process: %s', image_path)
                    # update current image in expert module
                    ins.update_image(image_path)
                    # call to expert module, all vectors will be calculated at this step
                    ins.expert_module()
                    # convert image to base64-encoded string, it is necessary to visualize
                    # in the main interface
                    is_success, buffer = cv2.imencode('.png', cv2.resize(ins._image, (256, 256)))
                    logger.debug('Image convertion code: %d', is_success)
                    b_io = BytesIO(buffer)
                    b_io.seek(0)
                    b64string = base64.b64encode(b_io.read())
                    # update image in the interface
                    window['face_image'].update(data=b64string, size=(256, 256))
            except Exception as exception:
                logger.error('Updating image error: %s', exception)
        # extracting facial features and update them if necessary
        elif event == 'Выделение признаков лица':
            # get current face vector from expert module
            face_vector = ins.get_facevector()
            logger.info('Extracted face vector: %s', face_vector)
            try:
                # layout for face features interface; initialize them here
                # because of the default values of classifiers predictions
                col1 = [[text_element_left('Пропорции лица')],
                        [text_element_left('Наличие бороды')],
                        [text_element_left('Выраженные скулы')],
                        [text_element_left('Толщина бровей')],
                        [text_element_left('Нос')],
                        [text_element_left('Цвет глаз')],
                        [text_element_left('Чёлка')],
                        [text_element_left('Цвет волос')],
                        [text_element_left('Наличие усов')],
                        [text_element_left('Бледность кожи')]]
                col2 = [[combo_element(key='ratio',
                                       values=('normal', 'wider', 'longer'),
                                       default_value=face_vector['ratio'])],
                        [combo_element(key='beard',
                                       values=('yes', 'no'),
                                       default_value=face_vector['beard'])],
                        [combo_element(key='highcheeckbones',
                                       values=('yes', 'no'),
                                       default_value=face_vector['highcheeckbones'])],
                        [combo_element(key='eyebrows_thickness',
                                       values=('thick', 'thin', 'normal'),
                                       default_value=face_vector['eyebrows_thickness'])],
                        [combo_element(key='nose_size',
                                       values=('big', 'long', 'small'),
                                       default_value=face_vector['nose_size'])],
                        [combo_element(key='eyes_iris',
                                       values=('brown', 'gray', 'green', 'blue'),
                                       default_value=face_vector['eyes_iris'])],
                        [combo_element(key='bangs', values=('yes', 'no'),
                                       default_value=face_vector['bangs'])],
                        [combo_element(key='hair',
                                       values=('black', 'brown', 'gray', 'blonde', 'red'),
                                       default_value=face_vector['hair'])],
                        [combo_element(key='mustache',
                                       values=('yes', 'no'),
                                       default_value=face_vector['mustache'])],
                        [combo_element(key='paleskin',
                                       values=('yes', 'no'),
                                       default_value=face_vector['paleskin'])]]
                col3 = [[text_element_right('Форма челюсти')],
                        [text_element_right('Двойной подбородок')],
                        [text_element_right('Полнота')],
                        [text_element_right('Форма бровей')],
                        [text_element_right('Узкие глаза')],
                        [text_element_right('Лоб')],
                        [text_element_right('Губы')],
                        [text_element_right('Наличие залысин')],
                        [text_element_right('Оттенок кожи')],
                        [text_element_right('Пол')]]
                col4 = [[combo_element(key='jawtype',
                                       values=('soft', 'angular'),
                                       default_value=face_vector['jawtype'])],
                        [combo_element(key='doublechin',
                                       values=('yes', 'no'),
                                       default_value=face_vector['doublechin'])],
                        [combo_element(key='chubby',
                                       values=('yes', 'no'),
                                       default_value=face_vector['chubby'])],
                        [combo_element(key='eyebrows_shape',
                                       values=('flat', 'curly', 'roof', 'angry'),
                                       default_value=face_vector['eyebrows_shape'])],
                        [combo_element(key='eyes_narrow',
                                       values=('yes', 'no'),
                                       default_value=face_vector['eyes_narrow'])],
                        [combo_element(key='forehead',
                                       values=('big', 'notbig'),
                                       default_value=face_vector['forehead'])],
                        [combo_element(key='lips',
                                       values=('big', 'normal'),
                                       default_value=face_vector['lips'])],
                        [combo_element(key='bald',
                                       values=('yes', 'no'),
                                       default_value=face_vector['bald'])],
                        [combo_element(key='skintone',
                                       values=('neutral', 'warm', 'cool'),
                                       default_value=face_vector['skintone'])],
                        [combo_element(key='gender',
                                       values=('female', 'male'),
                                       default_value=face_vector['gender'])]]
                cols = [col1, col2, col3, col4]
                facelayout = [[sg.Text('Редактирование лицевых признаков при необходимости',
                                       font=BIG_FONT)],
                              [sg.Column(col) for col in cols],
                              [button_element('Сохранить изменения')]]

                # create window for correcting images
                tmp_win = sg.Window('Лицевые признаки', facelayout, auto_size_text=True,
                                    grab_anywhere=True, font=SMALL_FONT).Finalize()
                face_events, face_values = tmp_win.read()
                tmp_win.close()
                logger.debug('Triggered face event: %s', face_events)
                logger.info('Changed face values: %s', face_values)
                for k, v in face_values.items():
                    if k in face_vector:
                        face_vector[k] = v
                # update face vector of corresponding image
                ins.update_facevector(face_vector)
            except Exception as exception:
                logger.error('Extracting and editing face vector error: %s', exception)
        # converting facial features to eyeglasses features
        elif event == 'Составление признаков оправы':
            ins.expert_module()
        # save already processed results
        elif event == 'Сохранить':
            ins.save()
        # eyeglasses recommendation's explanation module
        elif event == 'Вывод описания системы':
            try:
                descr = ins.description
            except AttributeError:
                logger.info('No description, creating new one...')
                ins.expert_module()
                descr = ins.description
            logger.info('Explanation of expert system: %s', descr)
            # layout for explanation window
            descrlayout = [[sg.Text('Описание рекомендательной системы', font=BIG_FONT)],
                           [button_element('Ок')],
                           [sg.Column([[sg.Text(descr, font=TINY_FONT, auto_size_text=True)]],
                                      scrollable=True)]]
            # layout for explanation window
            tmp_win = sg.Window('Описание экспертной рекомендации',
                                descrlayout, auto_size_text=True,
                                grab_anywhere=True, font=SMALL_FONT
                                ).Finalize()
            # Ok or close button pressed
            _ = tmp_win.read()
            tmp_win.close()
        # top-6 most suitable eyeglasses
        elif event == 'Подбор оправ по базе':
            ins.plot_recommendations(block=False)
        # unique eyeglasses with the help of GANs
        elif event == 'Сгенерировать индивидуальную оправу':
            ins.generate_unique(block=False)
        # exit button or other unexpected user behavior
        else:
            window.close()
            # breaking infinite loop, finishing program
            break
