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
    * English and russian localization for the interface

Example:
    To launch script, you should use the following command from the terminal:
        $ python gui.py
    In the first popup window you should load an image to process, i.e. the fast start,
    module would automatically calculate all the necessary stuff in background. After
    that all machine learning models will be loaded in RAM (loading usually takes from 30
    seconds up to 2 minutes). When progressbar will be filled, the main interface will appear.

Todo:
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
import json

import PySimpleGUI as sg
import cv2

from expert_backend import ExpertEyeglassesRecommender

VERSION = __version__ = '0.2.5 Released 27-August-2020'

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
    logger.setLevel(logging.DEBUG)

    # define file handler and set formatter
    file_handler = logging.FileHandler('logfile.log')
    file_handler.setLevel(logging.DEBUG)

    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)

    # layout for start screen
    layout0 = [[sg.Column([[sg.Text('Choose an image with face', key='chose_text')],
                           [sg.Input(key='file', visible=False, enable_events=True),
                            sg.FileBrowse('Open', key='open')],
                           [sg.Text('Language', key='lang_text')],
                           [combo_element(key='lang', values=('English', 'Русский'),
                                          default_value='English', enable_events=True)]],
                          element_justification='center')]]

    # create the Window
    start_win = sg.Window('Start window', layout0)
    while True:
        _, values = start_win.read()

        if values['lang'] == 'English':
            LANGUAGE = 'en'
        elif values['lang'] == 'Русский':
            LANGUAGE = 'ru'

        # load localization of GUI
        with open("lang/%s_lang.json" % LANGUAGE, "r") as f:
            lang_gui = json.load(f)['gui'].copy()

        # update text in the current window
        start_win['lang_text'].update(lang_gui['layout0']['lang_text'])
        start_win['chose_text'].update(lang_gui['layout0']['chose'])
        start_win['open'].update(lang_gui['layout0']['open'])
        start_win.TKroot.title(lang_gui['start_win']['title'])

        # if file was chosen, stop the loop
        if values['file']:
            break

    # need to destroy the window as it's still open
    start_win.close()

    # layout for loading components Window
    layout1 = [[sg.Text(lang_gui['layout1']['header'])],
               [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progbar')]]

    image_path = values['file']
    # check when no image was chosen
    if not image_path:
        logger.error(lang_gui['logger']['error']['no_image'])
        sys.exit()

    # window for loading of all necessary components
    prog_win = sg.Window(lang_gui['prog_win']['title'], layout1)
    logger.info(lang_gui['logger']['info']['image'], image_path)

    # initializing window and progressbar
    event, values = prog_win.read(timeout=10)
    prog_win['progbar'].update_bar(0)

    # create instance of ExpertEyeglassesRecommender with window callback to update progressbar
    ins = ExpertEyeglassesRecommender(image_path, prog_win, logger, lang=LANGUAGE)
    prog_win['progbar'].update_bar(100)

    # need to destroy the window as it's still open
    prog_win.close()

    # convert image to base64-encoded string, it is necessary to visualize in the main interface
    is_success, buffer = cv2.imencode('.png', cv2.resize(ins._image, (256, 256)))
    logger.debug(lang_gui['logger']['debug']['base64'], is_success)
    b_io = BytesIO(buffer)
    b_io.seek(0)
    b64string = base64.b64encode(b_io.read())

    # layout for main window
    layout2 = [[sg.Column([[sg.Text(lang_gui['layout2']['header'], font=BIG_FONT)],
                           [sg.Input(visible=False, enable_events=True, key='file'),
                            sg.FileBrowse(lang_gui['layout2']['update']),
                            button_element(lang_gui['layout2']['extract'], key='extract')],
                           [button_element(lang_gui['layout2']['translate'], key='translate'),
                            button_element(lang_gui['layout2']['explain'], key='explain')],
                           [sg.Image(data=b64string, size=(256, 256),
                                     pad=(64, 64), key='face_image')],
                           [button_element(lang_gui['layout2']['save'], key='save'),
                            button_element(lang_gui['layout2']['recommend'], key='recommend'),
                            button_element(lang_gui['layout2']['generate'], key='generate')]])]]

    # creating main interface
    main_win = sg.Window(lang_gui['main_win']['title'],
                         layout2, resizable=True,
                         grab_anywhere=True, font=SMALL_FONT)

    # graphical interface work in synchronous mode so we need to wait
    # for each event or command in infinite loop
    while True:
        event, values = main_win.read()
        # updating image scenario
        if event == 'file':
            try:
                image_path = values['file']
                # checking if no file was chosen
                if image_path is not None and image_path != '':
                    logger.info(lang_gui['logger']['info']['process'], image_path)
                    # update current image in expert module
                    ins.update_image(image_path)
                    # call to expert module, all vectors will be calculated at this step
                    ins.expert_module()
                    # convert image to base64-encoded string, it is necessary to visualize
                    # in the main interface
                    is_success, buffer = cv2.imencode('.png', cv2.resize(ins._image, (256, 256)))
                    logger.debug(lang_gui['logger']['debug']['base64'], is_success)
                    b_io = BytesIO(buffer)
                    b_io.seek(0)
                    b64string = base64.b64encode(b_io.read())
                    # update image in the interface
                    main_win['face_image'].update(data=b64string, size=(256, 256))
            except Exception as exception:
                logger.error(lang_gui['logger']['error']['update_image'], exception)
        # extracting facial features and update them if necessary
        elif event == 'extract':
            # get current face vector from expert module
            face_vector = ins.get_facevector()
            logger.info(lang_gui['logger']['info']['extract'], face_vector)
            try:
                # layout for face features interface; initialize them here
                # because of the default values of classifiers predictions
                col1 = [[text_element_left(lang_gui['facelayout']['col1']['proprotions'])],
                        [text_element_left(lang_gui['facelayout']['col1']['beard'])],
                        [text_element_left(lang_gui['facelayout']['col1']['cheeckbones'])],
                        [text_element_left(lang_gui['facelayout']['col1']['eyebrowthick'])],
                        [text_element_left(lang_gui['facelayout']['col1']['nose'])],
                        [text_element_left(lang_gui['facelayout']['col1']['eyes'])],
                        [text_element_left(lang_gui['facelayout']['col1']['bangs'])],
                        [text_element_left(lang_gui['facelayout']['col1']['haircolor'])],
                        [text_element_left(lang_gui['facelayout']['col1']['mustache'])],
                        [text_element_left(lang_gui['facelayout']['col1']['paleskin'])]]
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
                                       values=('black', 'brown', 'grey', 'blonde', 'red'),
                                       default_value=face_vector['hair'])],
                        [combo_element(key='mustache',
                                       values=('yes', 'no'),
                                       default_value=face_vector['mustache'])],
                        [combo_element(key='paleskin',
                                       values=('yes', 'no'),
                                       default_value=face_vector['paleskin'])]]
                col3 = [[text_element_right(lang_gui['facelayout']['col3']['jawtype'])],
                        [text_element_right(lang_gui['facelayout']['col3']['doublechin'])],
                        [text_element_right(lang_gui['facelayout']['col3']['chubby'])],
                        [text_element_right(lang_gui['facelayout']['col3']['eyebrowshape'])],
                        [text_element_right(lang_gui['facelayout']['col3']['narrow_eyes'])],
                        [text_element_right(lang_gui['facelayout']['col3']['forehead'])],
                        [text_element_right(lang_gui['facelayout']['col3']['lips'])],
                        [text_element_right(lang_gui['facelayout']['col3']['bald'])],
                        [text_element_right(lang_gui['facelayout']['col3']['skintone'])],
                        [text_element_right(lang_gui['facelayout']['col3']['sex'])]]
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
                facelayout = [[sg.Text(lang_gui['facelayout']['header'],
                                       font=BIG_FONT)],
                              [sg.Column(col) for col in cols],
                              [button_element(lang_gui['facelayout']['save'])]]

                # create window for correcting images
                face_win = sg.Window(lang_gui['face_win']['title'], facelayout,
                                     auto_size_text=True, grab_anywhere=True, font=SMALL_FONT
                                    ).Finalize()
                face_events, face_values = face_win.read()
                face_win.close()
                logger.debug(lang_gui['logger']['debug']['face_events'], face_events)
                logger.info(lang_gui['logger']['info']['new_face'], face_values)
                for k, v in face_values.items():
                    if k in face_vector:
                        face_vector[k] = v
                # update face vector of corresponding image
                ins.update_facevector(face_vector)
            except Exception as exception:
                logger.error(lang_gui['logger']['error']['extracting_editing'], exception)
        # converting facial features to eyeglasses features
        elif event == 'translate':
            ins.expert_module()
        # save already processed results
        elif event == 'save':
            ins.save()
        # eyeglasses recommendation's explanation module
        elif event == 'explain':
            try:
                descr = ins.description
            except AttributeError:
                logger.info(lang_gui['logger']['info']['no_description'])
                ins.expert_module()
                descr = ins.description
            logger.info(lang_gui['logger']['info']['explanation'], descr)
            # layout for explanation window
            descrlayout = [[sg.Text(lang_gui['descrlayout']['header'], font=BIG_FONT)],
                           [button_element(lang_gui['descrlayout']['ok'])],
                           [sg.Column([[sg.Text(descr, font=TINY_FONT, auto_size_text=True)]],
                                      scrollable=True)]]
            # layout for explanation window
            explain_win = sg.Window(lang_gui['explain_win']['title'],
                                    descrlayout, auto_size_text=True,
                                    grab_anywhere=True, font=SMALL_FONT).Finalize()
            # Ok or close button pressed
            _ = explain_win.read()
            explain_win.close()
        # top-6 most suitable eyeglasses
        elif event == 'recommend':
            ins.plot_recommendations(block=False)
        # unique eyeglasses with the help of GANs
        elif event == 'generate':
            ins.generate_unique(block=False)
        # exit button or other unexpected user behavior
        else:
            main_win.close()
            # breaking infinite loop, finishing program
            break
