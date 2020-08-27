#!/usr/bin/python3
'''Core of Expert eyeglasses recommendation system.

In this module the translation from facial feature to eyeglasses was implemented.
The task of recommending eyeframes based on people's appearance is quite hard
and there are no similar projects available at the moment. Facial features
consists of more than 20 facial attributes, with the help of expert module
function apply written beforehand rules to get necessary mappings. High
interpretability of such approach guaranties user's understanding and loyality
towards the system.

At the current state in translation function was already implemented:
    * Applying more than 40 branches of rules
    * Processing the scenario, when there is no dominant face shape
    * Paying more attention to other face shapes when dominant face shape is oval/oblong
    * Static-based rules, written in code
    * Description of applied rules in english/russian for great interpretability

Example:
    To use this module, you simply import class in your python code:
        # from expert_and_explanation import translate_facevec2eyeglassesvec

    After that you can call it with given protocol

Todo:
    * Place description's texts in separate file
    * Place rules in separate file, so it will be possible to manually change them
    * Implementation of summarizing results of explanation module
    * Rewrite code so it will be more scalable with raising number of rules

.. _Expert eyeglasses recommendation system with Generative Adversarial Networks:
   https://github.com/Defasium/expertglasses

'''
import json

import numpy as np

VERSION = __version__ = '0.2.5 Released 27-August-2020'

def translate_facevec2eyeglassesvec(facevector: dict, s_vector: np.ndarray, c_vector: np.ndarray,
                                    lang='ru'):
    '''Maps facial features to eyeglasses features
    by consequentially applying static expert rules. Also
    constructs description for each rule, which can be used to
    interpret the results of the translation.

            Args:
                facevector (dict with strings as key): Dictionary with facial attributes.
                s_vector (numpy.ndarray): Initial shape attributes of eyeglasses.
                c_vector (numpy.ndarray): Initial color attributes of eyeglasses.
                lang (str, default='ru'): What language to use for description.
                Supported values are ['ru', 'en'].
            Returns:
                s_vector (numpy.ndarray): Final shape attributes of eyeglasses.
                c_vector (numpy.ndarray): Final color attributes of eyeglasses.
                description (str): Explanation of translation and applied rules.

    '''

    # load specified localization from json
    with open('lang/%s_lang.json' % lang, 'r') as lang_file:
        lang_i = json.load(lang_file)['explanation'].copy()

    description = []

    ##############################
    ######### Faceshape ##########
    ##############################
    description.append(lang_i['faceshape']['header'])
    if facevector['faceshape'][0][0] < 0.35:
        emphasize_coef = 3
    else:
        emphasize_coef = 1
    condition = facevector['faceshape'][0][1] in ['oval', 'oblong']
    for fraction, faceshape in facevector['faceshape']:
        if fraction < 0.05:
            continue
        # more attention to second component
        if condition and faceshape == facevector['faceshape'][1][1]:
            attention_on_other_shapes = 3
        else:
            attention_on_other_shapes = 1
        if faceshape == 'oval':
            s_vector[[0, 1, 4, 8]] += fraction * 2
            s_vector[[2, 3, 5, 6, 7, 9]] += fraction / 2
            s_vector[14] += fraction  # small and oversized are bad choice
            s_vector[[13, 15, 16, 18]] -= fraction
            if facevector['ratio'] == 'wider':
                s_vector[12] += fraction / len(facevector['faceshape'])
            else:
                s_vector[[10, 11]] += fraction
            c_vector += fraction
        elif faceshape == 'triangle':
            s_vector[[3, 5, 7]] += fraction * 2 * attention_on_other_shapes
            s_vector[[0, 1, 2, 4]] += fraction / 2 * attention_on_other_shapes
            s_vector[[28]] += fraction * attention_on_other_shapes
            s_vector[[29]] += fraction * attention_on_other_shapes / 2
            c_vector[[0]] -= fraction * attention_on_other_shapes
            c_vector[[3, 8, 12]] += fraction * attention_on_other_shapes
            c_vector[[9]] += fraction * attention_on_other_shapes / 3
        elif faceshape == 'oblong':
            s_vector[[3, 4]] += fraction * 2
            s_vector[[2]] += fraction / 2
            s_vector[14] += fraction
            s_vector[15] += fraction * 2
            s_vector[28] += fraction / 2
            c_vector[13] += fraction
            c_vector[14] += fraction
        elif faceshape == 'heart':
            s_vector[[2, 4, 7, 9]] += fraction * 2 * attention_on_other_shapes
            s_vector[[0, 1, 3]] += fraction / 2 * attention_on_other_shapes
            s_vector[[11, 12]] += fraction * attention_on_other_shapes
            s_vector[[10]] += fraction / 2 * attention_on_other_shapes
            s_vector[20] -= fraction * 2 * attention_on_other_shapes
            s_vector[22] += fraction * attention_on_other_shapes
            c_vector[[1, 8]] += fraction * attention_on_other_shapes
            c_vector[[9]] += fraction / 3 * attention_on_other_shapes
        elif faceshape == 'diamond':
            s_vector[[2, 4, 7]] += fraction * attention_on_other_shapes
            s_vector[[29]] += fraction * attention_on_other_shapes
            s_vector[[11]] += fraction * attention_on_other_shapes
            s_vector[20] += fraction * attention_on_other_shapes
            # small and oversized are bad choice
            s_vector[14] += fraction * attention_on_other_shapes
            s_vector[[13, 15, 16, 18]] -= fraction * attention_on_other_shapes
            c_vector[[1, 2, 13]] += fraction * attention_on_other_shapes
        elif faceshape == 'round':
            s_vector[[0, 1, 5, 9]] += fraction * 2 * attention_on_other_shapes
            s_vector[[2, 3]] -= fraction * 2 * attention_on_other_shapes
            s_vector[14] += fraction * attention_on_other_shapes
            s_vector[[13, 15, 16, 18]] -= fraction * attention_on_other_shapes
            s_vector[15] += fraction / 2 * attention_on_other_shapes
            s_vector[19] += fraction / 2 * attention_on_other_shapes
            s_vector[[10, 11]] += fraction * 2 * attention_on_other_shapes
            s_vector[[22]] -= fraction * 2 * attention_on_other_shapes
            c_vector[[1, 2, 13]] += fraction * attention_on_other_shapes
        elif faceshape == 'square':
            s_vector[[3]] += fraction * 2 * attention_on_other_shapes
            s_vector[[2, 5]] += fraction * attention_on_other_shapes
            s_vector[[0, 1]] -= fraction * 2 * attention_on_other_shapes
            s_vector[[17, 18]] += fraction * attention_on_other_shapes
            s_vector[[28]] += fraction / 2 * attention_on_other_shapes
            s_vector[[21]] -= fraction * 2 * attention_on_other_shapes
            s_vector[[22]] -= fraction * attention_on_other_shapes
            c_vector[[10, 3, 5, 7, 12]] += fraction * attention_on_other_shapes
            c_vector[[9]] += fraction / 3 * attention_on_other_shapes
        elif faceshape == 'rectangle':
            s_vector[[3, 4]] += fraction * attention_on_other_shapes
            s_vector[[2]] += fraction / 2 * attention_on_other_shapes
            s_vector[14] += fraction / 2 * attention_on_other_shapes
            s_vector[15] += fraction * attention_on_other_shapes
            c_vector[13] += fraction * attention_on_other_shapes
            c_vector[14] += fraction * attention_on_other_shapes
            s_vector[[11, 12]] += fraction / 2 * attention_on_other_shapes

        # add description
        description.append(lang_i['faceshape'][faceshape]['prefix'])
        description.append(f'{fraction*100:.1f}%\n')
        description.append(lang_i['faceshape'][faceshape]['description'])

    ##############################
    ######### Faceratio ##########
    ##############################
    description.append(lang_i['faceratio']['header'])
    if facevector['ratio'] == 'wider':
        s_vector[[10, 20]] += 2.0 * emphasize_coef
        s_vector[[15, 16]] -= 2.0 * emphasize_coef
    elif facevector['ratio'] == 'longer':
        s_vector[[20]] -= 2.0 * emphasize_coef
        s_vector[[14, 15]] += 2.0 * emphasize_coef

    description.append(lang_i['faceratio'][facevector['ratio']]['prefix'])
    description.append(lang_i['faceratio'][facevector['ratio']]['description'])

    ##############################
    ########## Jawtype ###########
    ##############################
    description.append(lang_i['jawtype']['header'])
    if facevector['beard'] == 'no':  # huge beard biases focus from faceshape
        if facevector['jawtype'] == 'soft' or \
                        facevector['doublechin'] == 'yes' or \
                        facevector['chubby'] == 'yes':
            description.append(lang_i['jawtype']['soft']['prefix'])
            description.append(lang_i['jawtype']['soft']['description'])
            s_vector[[21]] += 2.0 * emphasize_coef
            s_vector[[2, 3, 5]] += 2.0 * emphasize_coef
        else:
            description.append(lang_i['jawtype']['defined']['prefix'])
            description.append(lang_i['jawtype']['defined']['description'])
            s_vector[[21]] -= 2.0 * emphasize_coef
            s_vector[[0, 1]] += 2.0 * emphasize_coef
    else:
        description.append(lang_i['jawtype']['beard']['prefix'])
        description.append(lang_i['jawtype']['beard']['description'])
        s_vector[[20]] += 1.0 * emphasize_coef

    ##############################
    ######### Eyebrows ###########
    ##############################
    description.append(lang_i['eyebrows']['header'])
    if facevector['eyebrows_thickness'] == 'thick':
        description.append(lang_i['eyebrows'][facevector['eyebrows_thickness']]['prefix'])
        description.append(lang_i['eyebrows'][facevector['eyebrows_thickness']]['description'])
        s_vector[[11, 12]] += 1.0 * emphasize_coef
        s_vector[[22]] -= 1.0 * emphasize_coef
    elif facevector['eyebrows_thickness'] == 'thin':
        description.append(lang_i['eyebrows'][facevector['eyebrows_thickness']]['prefix'])
        description.append(lang_i['eyebrows'][facevector['eyebrows_thickness']]['description'])
        s_vector[[10]] += 1.0 * emphasize_coef
        s_vector[[22]] += 1.0 * emphasize_coef

    if facevector['eyebrows_shape'] == 'flat':
        s_vector[[23]] += 2.0 * emphasize_coef
    elif facevector['eyebrows_shape'] == 'curly':
        s_vector[[24, 25]] += 2.0 * emphasize_coef
    elif facevector['eyebrows_shape'] == 'roof':
        s_vector[[26]] += 2.0 * emphasize_coef
    elif facevector['eyebrows_shape'] == 'angry':
        s_vector[[27]] += 2.0 * emphasize_coef

    description.append(lang_i['eyebrows'][facevector['eyebrows_shape']]['prefix'])
    description.append(lang_i['eyebrows'][facevector['eyebrows_shape']]['description'])

    ##############################
    ############ Nose ############
    ##############################
    description.append(lang_i['nose']['header'])
    if facevector['nose_size'] in ('big', 'long'):
        s_vector[[10]] += 1.0 * emphasize_coef
        s_vector[[22]] += 1.0 * emphasize_coef
        s_vector[[19]] -= 1.0 * emphasize_coef
    else:
        s_vector[[11, 12]] += 0.25
        s_vector[[19]] += 1.0

    description.append(lang_i['nose'][facevector['nose_size']]['prefix'])
    description.append(lang_i['nose'][facevector['nose_size']]['description'])

    ##############################
    ############ Eyes ############
    ##############################
    description.append(lang_i['eyes']['header'])
    if facevector['eyes_narrow'] == 'yes':
        description.append(lang_i['eyes']['narrow']['prefix'])
        description.append(lang_i['eyes']['narrow']['description'])
        s_vector[[14, 15]] += 1.0 * emphasize_coef
        s_vector[[17, 18]] += 1.0 * emphasize_coef  # narrow eyes -> larger frames
        c_vector[[0, 1]] -= 1.0 * emphasize_coef
        s_vector[[4, 8]] += 0.5 * emphasize_coef
        s_vector[[22]] -= 0.5 * emphasize_coef
        c_vector[[9]] += 0.3

    if facevector['eyes_iris'] == 'brown':
        c_vector[[1, 3, 12, 13]] += 1.0
        c_vector[[9]] += 0.3
    elif facevector['eyes_iris'] == 'blue':
        c_vector[[2, 3, 5, 12]] += 1.0
        c_vector[[9]] += 0.3
    elif facevector['eyes_iris'] == 'gray':  # anything
        c_vector += 1.0
    elif facevector['eyes_iris'] == 'green':
        c_vector[[1, 5, 6, 10, 7, 12]] += 1.0
        c_vector[[9]] += 0.3

    description.append(lang_i['eyes'][facevector['eyes_iris']]['prefix'])
    description.append(lang_i['eyes'][facevector['eyes_iris']]['description'])

    ##############################
    ######### Forehead ###########
    ##############################
    description.append(lang_i['forehead']['header'])
    if facevector['forehead'] == 'big' and facevector['bangs'] == 'no':
        description.append(lang_i['forehead']['big']['prefix'])
        description.append(lang_i['forehead']['big']['description'])
        s_vector[[22]] += 2.0 * emphasize_coef
        s_vector[[10]] += 2.0 * emphasize_coef
    else:
        description.append(lang_i['forehead']['small']['prefix'])
        description.append(lang_i['forehead']['small']['description'])
        s_vector[[22]] -= 0.25 * emphasize_coef
        s_vector[[11, 12]] += 0.25 * emphasize_coef

    ##############################
    ########### Lips #############
    ##############################
    description.append(lang_i['lips']['header'])
    if facevector['lips'] == 'big' and facevector['mustache'] == 'no':
        description.append(lang_i['lips'][facevector['lips']]['prefix'])
        description.append(lang_i['lips'][facevector['lips']]['description'])
        s_vector[[22]] += 0.5 * emphasize_coef
        s_vector[[10]] += 0.5 * emphasize_coef
    else:
        description.append(lang_i['lips'][facevector['lips']]['prefix'])
        description.append(lang_i['lips'][facevector['lips']]['description'])
        s_vector[[22]] -= 0.125 * emphasize_coef
        s_vector[[11, 12]] += 0.125 * emphasize_coef

    ##############################
    ########## Baldness ##########
    ##############################
    description.append(lang_i['hair']['header'])
    if facevector['bald'] == 'yes':
        description.append(lang_i['hair']['bald']['prefix'])
        description.append(lang_i['hair']['bald']['description'])
        s_vector[[23]] += 2.0 * emphasize_coef
        s_vector[[22]] += 2.0 * emphasize_coef
        s_vector[[10]] += 1.0 * emphasize_coef
        c_vector[0] += 1.0

    ##############################
    ######### Hair color #########
    ##############################
    if facevector['hair'] == 'black':
        s_vector[29] += 1.0
        s_vector[28] += 0.5
        c_vector[[2, 4, 10, 0, 1, 8, 3, 12, 9]] += 1.0
    elif facevector['hair'] == 'blonde':
        s_vector[28] += 1.0
        s_vector[29] += 0.5
        c_vector[[2, 13, 5, 10, 6, 0, 12]] += 1.0
    elif facevector['hair'] == 'brown':
        s_vector[29] += 1.0
        s_vector[28] += 0.5
        c_vector[[4, 13, 8, 5, 6, 7, 1, 0]] += 1.0
    elif facevector['hair'] == 'grey':
        s_vector[29] += 1.0
        s_vector[28] += 0.5
        c_vector[[11, 5, 1, 3, 7, 12, 0, 8]] += 1.0
    elif facevector['hair'] == 'red':
        c_vector[[2, 6, 5, 1, 12, 3, 0]] += 1.0

    description.append(lang_i['hair']['haircolor'][facevector['hair']]['prefix'])
    description.append(lang_i['hair']['haircolor'][facevector['hair']]['description'])

    ##############################
    ######### Skintone ###########
    ##############################
    description.append(lang_i['skintone']['header'])
    if facevector['skintone'] == 'warm':
        c_vector[[2, 5, 12, 1, 6]] += 1.0
    elif facevector['skintone'] == 'neutral':  # any color is good
        c_vector[[1, 5, 3, 6, 9, 7, 12, 0, 8]] += 1.0
        s_vector[29] += 1.0
    elif facevector['skintone'] == 'cool':
        c_vector[[4, 5, 3, 7, 6, 10, 9, 12, 0]] += 1.0

    description.append(lang_i['skintone'][facevector['skintone']]['prefix'])
    description.append(lang_i['skintone'][facevector['skintone']]['description'])

    if facevector['race'] == 'black':
        description.append(lang_i['skintone']['black']['prefix'])
        description.append(lang_i['skintone']['black']['description'])
        c_vector[[9, 12, 0, 4, 2, 11]] += 1.0
    elif facevector['paleskin'] == 'yes':
        description.append(lang_i['skintone']['pale']['prefix'])
        description.append(lang_i['skintone']['pale']['description'])
        c_vector[[9, 1, 0]] += 1.0
        s_vector[[10]] += 1.0 * emphasize_coef
        s_vector[[11]] += 0.5 * emphasize_coef
    else:
        c_vector[[0, 3, 6, 13]] += 0.5

    ##############################
    ############ Sex #############
    ##############################
    description.append(lang_i['sex']['header'])
    if facevector['gender'] == 'female':
        s_vector[[28]] += 2.0 * emphasize_coef
        s_vector[[10]] += 2.0 * emphasize_coef
        s_vector[[22]] += 2.0 * emphasize_coef
        s_vector[[30]] -= 1.0
    else:
        s_vector[[29]] += 1.0 * emphasize_coef
        s_vector[[10]] += 0.5 * emphasize_coef
        s_vector[[11]] += 1.0 * emphasize_coef
        s_vector[[30]] += 1.0

    description.append(lang_i['sex'][facevector['gender']]['prefix'])
    description.append(lang_i['sex'][facevector['gender']]['description'])

    return s_vector, c_vector, ''.join(description)
