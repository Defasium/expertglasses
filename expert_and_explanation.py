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
    * Description of applied rules in russian for great interpretability

Example:
    To use this module, you simply import class in your python code:
        # from expert_and_explanation import translate_facevec2eyeglassesvec

    After that you can call it with given protocol

Todo:
    * Add english version of description
    * Place description's texts in separate file
    * Place rules in separate file, so it will be possible to manually change them
    * Implementation of summarizing results of explanation module
    * Rewrite code so it will be more scalable with raising number of rules

.. _Expert eyeglasses recommendation system with Generative Adversarial Networks:
   https://github.com/Defasium/expertglasses

'''

import numpy as np

VERSION = __version__ = '0.2.0 Released 26-May-2020'

def translate_facevec2eyeglassesvec(facevector: dict, s_vector: np.ndarray, c_vector: np.ndarray):
    '''Maps facial features to eyeglasses features
    by consequentially applying static expert rules. Also
    constructs description for each rule, which can be used to
    interpret the results of the translation.

            Args:
                facevector (dict with strings as key): Dictionary with facial attributes.
                s_vector (numpy.ndarray): Initial shape attributes of eyeglasses
                c_vector (numpy.ndarray): Initial color attributes of eyeglasses
            Returns:
                s_vector (numpy.ndarray): Final shape attributes of eyeglasses.
                c_vector (numpy.ndarray): Final color attributes of eyeglasses.
                description (str): Explanation of translation and applied rules.

    '''
    description = ''

    ##############################
    ######### Faceshape ##########
    ##############################
    description += 'Форма лица:\n'
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
            description += f'\tОвальная на {fraction*100:.1f}%\n'
            description += '''\tТонкие оправы не лучший выбор (делают визуально шире),
             в основном хороши все. Стоит обратить внимание на оправы квадратной формы, 
            прямоугольной, авиаторы. Лучше избегать слишком больших и слишком мелких оправ.
            Ободковые и полуободковые оправы - хороший выбор для вытянутого лица.
            Любой цвет вам к лицу.\n'''
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
            description += f'\tТреугольная на {fraction*100:.1f}%\n'
            description += '''\tСтоит обратить внимание на оправы формы кошачий глаз,
             клабмастеры, круглой. Тонкие металлические оправы или пластиковые - вам к лицу.
            Лучше избегать оправ темных оттенков. Подходят легкие, светлые цвета. Также 
            рассмотрите вариант прозрачных пластиковых оправ.\n'''
            s_vector[[3, 5, 7]] += fraction * 2 * attention_on_other_shapes
            s_vector[[0, 1, 2, 4]] += fraction / 2 * attention_on_other_shapes
            s_vector[[28]] += fraction * attention_on_other_shapes
            s_vector[[29]] += fraction * attention_on_other_shapes / 2
            c_vector[[0]] -= fraction * attention_on_other_shapes
            c_vector[[3, 8, 12]] += fraction * attention_on_other_shapes
            c_vector[[9]] += fraction * attention_on_other_shapes / 3
        elif faceshape == 'oblong':
            description += f'\tВытянутая на {fraction*100:.1f}%\n'
            description += '''\tСтоит обратить внимание на оправы круглой формы и
             авиаторы. Для большей гармоничности рекомендуется рассмотреть оправы, вытянутые 
            в высоту, нежели чем в ширину. Попробуйте оправы, у котрых цвет дужек отличается 
            от основного цвета.\n'''
            s_vector[[3, 4]] += fraction * 2
            s_vector[[2]] += fraction / 2
            s_vector[14] += fraction
            s_vector[15] += fraction * 2
            s_vector[28] += fraction / 2
            c_vector[13] += fraction
            c_vector[14] += fraction
        elif faceshape == 'heart':
            description += f'\tСердцевидная на {fraction*100:.1f}%\n'
            description += '''\tУ вас заострённый подбородок. Для гармонии лучше выбирать
             оправы, расширающиеся книзу. Поэкспериментируйте со стилем ретро. Стоит обратить 
            внимание на оправы овальной формы, клабмастеры, авиаторы. Лучше избегать 
            сужающихся и очень тонких моделей. Подходят легкие, светлые цвета, 
            оттенки коричневого.\n'''
            s_vector[[2, 4, 7, 9]] += fraction * 2 * attention_on_other_shapes
            s_vector[[0, 1, 3]] += fraction / 2 * attention_on_other_shapes
            s_vector[[11, 12]] += fraction * attention_on_other_shapes
            s_vector[[10]] += fraction / 2 * attention_on_other_shapes
            s_vector[20] -= fraction * 2 * attention_on_other_shapes
            s_vector[22] += fraction * attention_on_other_shapes
            c_vector[[1, 8]] += fraction * attention_on_other_shapes
            c_vector[[9]] += fraction / 3 * attention_on_other_shapes
        elif faceshape == 'diamond':
            description += f'\tРомбовидная на {fraction*100:.1f}%\n'
            description += '''\tФорма лица редкая и сбалансированная. Чтобы подчеркнуть те
             или иные черты, рассмотрите оправы формы кошачий гла, клабмастеры, овальные. 
             Лучше избегать слишком больших и слишком мелких оправ. Лучше избегать 
             расширяющихся книзу моделей. Рассмотрите вариант металлических полуободковых 
             оправ. Подходят оттенки коричневого, черного. Оправы с темным верхом.\n'''
            s_vector[[2, 4, 7]] += fraction * attention_on_other_shapes
            s_vector[[29]] += fraction * attention_on_other_shapes
            s_vector[[11]] += fraction * attention_on_other_shapes
            s_vector[20] += fraction * attention_on_other_shapes
            # small and oversized are bad choice
            s_vector[14] += fraction * attention_on_other_shapes
            s_vector[[13, 15, 16, 18]] -= fraction * attention_on_other_shapes
            c_vector[[1, 2, 13]] += fraction * attention_on_other_shapes
        elif faceshape == 'round':
            description += f'\tКруглая на {fraction*100:.1f}%\n'
            description += '''\tСтоит обратить внимание на оправы формы кошачий глаз,
             прямоугольные, квадратные и в целом угловатые. Лучше избегать круглых и овальных
            оправ, невытянутых, а также толстых. Рассмотрите вариант моделей ободковых и 
            полуободковых. Подходят оттенки коричневого, черного. Оправы с принтом.\n'''
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
            description += f'\tКвадратная на {fraction*100:.1f}%\n'
            description += '''\tСтоит обратить внимание на оправы круглой, овальной формы,
             и в со сглаженными краями. Лучше избегать квардратных, прямоугольных, акцентирующих
            внимание на челюсти моделей. Рассмотрите вариант широкихх пластиковых оправ.
            Поэкспериментируйте с цветами, попробуйте яркие цвета.\n'''
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
            description += f'\tПрямоугольная на {fraction*100:.1f}%\n'
            description += '''\tСтоит обратить внимание на оправы круглой формы,
             авиаторы, оправы со сглаженными краями. Для большей гармоничности рекомендуется 
            рассмотреть оправы, вытянутые в высоту, нежели чем в ширину. Попробуйте оправы, 
            у котрых цвет дужек отличается от основного цвета.\n'''
            s_vector[[3, 4]] += fraction * attention_on_other_shapes
            s_vector[[2]] += fraction / 2 * attention_on_other_shapes
            s_vector[14] += fraction / 2 * attention_on_other_shapes
            s_vector[15] += fraction * attention_on_other_shapes
            c_vector[13] += fraction * attention_on_other_shapes
            c_vector[14] += fraction * attention_on_other_shapes
            s_vector[[11, 12]] += fraction / 2 * attention_on_other_shapes

    ##############################
    ######### Faceratio ##########
    ##############################
    description += 'Пропорции лица:\n'
    if facevector['ratio'] == 'wider':
        description += '''\tШире\n
        \tПодойдут оправы, вытянутые в ширину.\n'''
        s_vector[[10, 20]] += 2.0 * emphasize_coef
        s_vector[[15, 16]] -= 2.0 * emphasize_coef
    elif facevector['ratio'] == 'longer':
        description += '''\tВыше\n
        \tПодойдут оправы, вытянутые в высоту.\n'''
        s_vector[[20]] -= 2.0 * emphasize_coef
        s_vector[[14, 15]] += 2.0 * emphasize_coef

    ##############################
    ########## Jawtype ###########
    ##############################
    description += 'Форма челюсти:\n'
    if facevector['beard'] == 'no':  # huge beard biases focus from faceshape
        if facevector['jawtype'] == 'soft' or \
                        facevector['doublechin'] == 'yes' or \
                        facevector['chubby'] == 'yes':
            description += '''\tМягкая\n
            \tПодойдут оправы угловатой формы.\n'''
            s_vector[[21]] += 2.0 * emphasize_coef
            s_vector[[2, 3, 5]] += 2.0 * emphasize_coef
        else:
            description += '''\tВыраженная\n
            \tПодойдут оправы со сглаженными краями.\n'''
            s_vector[[21]] -= 2.0 * emphasize_coef
            s_vector[[0, 1]] += 2.0 * emphasize_coef
    else:
        description += '''\tПрисутствует борода\n
        \tБорода смещает акцент вниз, рекомендуется рассмотреть 
        сужающиеся книзу модели.\n'''
        s_vector[[20]] += 1.0 * emphasize_coef

    ##############################
    ######### Eyebrows ###########
    ##############################
    description += 'Брови:\n'
    if facevector['eyebrows_thickness'] == 'thick':
        description += '''\tТолстые\n
        \tОбодковые толстые оправы будут выглядеть с толстыми бровями комично.\n'''
        s_vector[[11, 12]] += 1.0 * emphasize_coef
        s_vector[[22]] -= 1.0 * emphasize_coef
    elif facevector['eyebrows_thickness'] == 'thin':
        description += '''\tТонкие\n
        \tОбодковые толстые оправы подчеркнут область глаз.\n'''
        s_vector[[10]] += 1.0 * emphasize_coef
        s_vector[[22]] += 1.0 * emphasize_coef

    if facevector['eyebrows_shape'] == 'flat':
        description += '''\tПлоская линия бровей\n'''
        s_vector[[23]] += 2.0 * emphasize_coef
    elif facevector['eyebrows_shape'] == 'curly':
        description += '''\tЗакруглённая линия бровей\n'''
        s_vector[[24, 25]] += 2.0 * emphasize_coef
    elif facevector['eyebrows_shape'] == 'roof':
        description += '''\tЛиния бровей <<домиком>>\n'''
        s_vector[[26]] += 2.0 * emphasize_coef
    elif facevector['eyebrows_shape'] == 'angry':
        description += '''\tХмурая линия бровей\n'''
        s_vector[[27]] += 2.0 * emphasize_coef
    description += '''\tФорма верхней части оправ должна повторять линию бровей\n'''
    ##############################
    ############ Nose ############
    ##############################
    description += 'Размер Носа:\n'
    if facevector['nose_size'] == 'big':
        description += '''\tКрупный\n
            Сместить внимание с крупного носа помогут оправы толстой формы.
            Отсутствие носоупоров сыграет на пользу.\n'''
        s_vector[[10]] += 1.0 * emphasize_coef
        s_vector[[22]] += 1.0 * emphasize_coef
        s_vector[[19]] -= 1.0 * emphasize_coef
    else:
        description += '''\tНебольшой\n
            Можно смело рассматривать модели с носоупорами.
            Полуобоковые и безободковые оправы стоят рассмотрения.\n'''
        s_vector[[11, 12]] += 0.25
        s_vector[[19]] += 1.0

    ##############################
    ############ Eyes ############
    ##############################
    description += 'Глаза:\n'
    if facevector['eyes_narrow'] == 'yes':
        description += '''\tУзкий разрез глаз:\n
            Для узкого разреза глаз лучше выбирать большие, не слишком толстые оправы.
            Темные оттенки - не лучший выбор.\n'''
        s_vector[[14, 15]] += 1.0 * emphasize_coef
        s_vector[[17, 18]] += 1.0 * emphasize_coef  # narrow eyes -> larger frames
        c_vector[[0, 1]] -= 1.0 * emphasize_coef
        s_vector[[4, 8]] += 0.5 * emphasize_coef
        s_vector[[22]] -= 0.5 * emphasize_coef
        c_vector[[9]] += 0.3

    if facevector['eyes_iris'] == 'brown':
        description += '''\tКарие\n
            Подходят модели коричневых, синих, темнозеленых, светлых оттенков.
            Также рассмотрите вариант прозрачных пластиковых оправ.\n'''
        c_vector[[1, 3, 12, 13]] += 1.0
        c_vector[[9]] += 0.3
    elif facevector['eyes_iris'] == 'blue':
        description += '''\tГолубые\n
            Подходят модели голубых, синих, оранжевых, красных оттенков.
            Также рассмотрите вариант прозрачных пластиковых оправ.\n'''
        c_vector[[2, 3, 5, 12]] += 1.0
        c_vector[[9]] += 0.3
    elif facevector['eyes_iris'] == 'gray':  # anything
        description += '''\tСерые\n
            Любые оттенки вам к лицу.\n'''
        c_vector += 1.0
    elif facevector['eyes_iris'] == 'green':
        description += '''\tЗелёные\n
            Яркие, светлые тона будут отлично смотреться.
            Также рассмотрите вариант прозрачных пластиковых оправ.\n'''
        c_vector[[1, 5, 6, 10, 7, 12]] += 1.0
        c_vector[[9]] += 0.3

    ##############################
    ######### Forehead ###########
    ##############################
    description += 'Лоб:\n'
    if facevector['forehead'] == 'big' and facevector['bangs'] == 'no':
        description += '''\tБольшой\n
            Сместить внимание с большого лба помогут толстые ободковые оправы.\n'''
        s_vector[[22]] += 2.0 * emphasize_coef
        s_vector[[10]] += 2.0 * emphasize_coef
    else:
        description += '''\tНебольшой или закрыт чёлкой\n
            Попробуйте не слишком толстые оправы.\n'''
        s_vector[[22]] -= 0.25 * emphasize_coef
        s_vector[[11, 12]] += 0.25 * emphasize_coef

    ##############################
    ########### Lips #############
    ##############################
    description += 'Губы:\n'
    if facevector['lips'] == 'big' and facevector['mustache'] == 'no':
        description += '''\tБольшие\n
            Сместить акцент с губ помогут толстые ободковые оправы.\n'''
        s_vector[[22]] += 0.5 * emphasize_coef
        s_vector[[10]] += 0.5 * emphasize_coef
    else:
        description += '''\tОбычные\n
            Попробуйте не слишком толстые оправы.\n'''
        s_vector[[22]] -= 0.125 * emphasize_coef
        s_vector[[11, 12]] += 0.125 * emphasize_coef

    ##############################
    ########## Baldness ##########
    ##############################
    description += 'Волосы:\n'
    if facevector['bald'] == 'yes':
        description += '''\tЕсть залысины или отсутствуют волосы\n
            Подойдут толстые черные оправы с плоской верхней частью.\n'''
        s_vector[[23]] += 2.0 * emphasize_coef
        s_vector[[22]] += 2.0 * emphasize_coef
        s_vector[[10]] += 1.0 * emphasize_coef
        c_vector[0] += 1.0

    ##############################
    ######### Hair color #########
    ##############################
    if facevector['hair'] == 'black':
        description += '''\tЧерные\n
            Металлические оправы с темными или блестящими оттенками.
            Также рассмотрите вариант прозрачных пластиковых оправ.\n'''
        s_vector[29] += 1.0
        s_vector[28] += 0.5
        c_vector[[2, 4, 10, 0, 1, 8, 3, 12, 9]] += 1.0
    elif facevector['hair'] == 'blonde':
        description += '''\tСветлые\n
            Подойдут яркие цвета, позолоченные оправы, белые оправы.\n'''
        s_vector[28] += 1.0
        s_vector[29] += 0.5
        c_vector[[2, 13, 5, 10, 6, 0, 12]] += 1.0
    elif facevector['hair'] == 'brown':
        description += '''\tКоричневые\n
            Подойдут яркие цвета, позолоченные оправы, белые оправы.\n'''
        s_vector[29] += 1.0
        s_vector[28] += 0.5
        c_vector[[4, 13, 8, 5, 6, 7, 1, 0]] += 1.0
    elif facevector['hair'] == 'grey':
        description += '''\tСедые\n
            Подойдут цвета метталик, черные, коричневые, фиолетовые.\n'''
        s_vector[29] += 1.0
        s_vector[28] += 0.5
        c_vector[[11, 5, 1, 3, 7, 12, 0, 8]] += 1.0
    elif facevector['hair'] == 'red':
        description += '''\tРыжие\n
            Подойдут яркие цвета: золотой, зелёный, красный, белый, синий.\n'''
        c_vector[[2, 6, 5, 1, 12, 3, 0]] += 1.0

    ##############################
    ######### Skintone ###########
    ##############################
    description += 'Тон кожи:\n'
    if facevector['skintone'] == 'warm':
        description += '''\tТеплый\n
            Подойдут теплые оттенки, золотые оправы.\n'''
        c_vector[[2, 5, 12, 1, 6]] += 1.0
    elif facevector['skintone'] == 'neutral':  # any color is good
        description += '''\tНейтральный\n
            Любые цвета вам к лицу.\n'''
        c_vector[[1, 5, 3, 6, 9, 7, 12, 0, 8]] += 1.0
        s_vector[29] += 1.0
    elif facevector['skintone'] == 'cool':
        description += '''\tХолодный\n
            Подойдут холодные оттенки, серебряные оправы.\n'''
        c_vector[[4, 5, 3, 7, 6, 10, 9, 12, 0]] += 1.0

    if facevector['race'] == 'black':
        description += '''\tТемная\n
            Отличной идеей будет сыграть на контрастах: светлые и прозрачные модели.\n'''
        c_vector[[9, 12, 0, 4, 2, 11]] += 1.0
    elif facevector['paleskin'] == 'yes':
        description += '''\tБледная\n
            Отличной идеей будет сыграть на контрастах: черные, коричневые модели.
            Также рассмотрите вариант прозрачных пластиковых оправ.
            Безободковые оправы - не лучший выбор. В них вы будете выглядеть старше.\n'''
        c_vector[[9, 1, 0]] += 1.0
        s_vector[[10]] += 1.0 * emphasize_coef
        s_vector[[11]] += 0.5 * emphasize_coef
    else:
        c_vector[[0, 3, 6, 13]] += 0.5

    ##############################
    ########## Gender ############
    ##############################
    description += 'Пол:\n'
    if facevector['gender'] == 'female':
        description += '''\tЖенский\n
            Согласно социологическому исследованию, девушки отдают предпочтение
            ободковым толстым пластиковым оправам.\n'''
        s_vector[[28]] += 2.0 * emphasize_coef
        s_vector[[10]] += 2.0 * emphasize_coef
        s_vector[[22]] += 2.0 * emphasize_coef
        s_vector[[30]] -= 1.0
    else:
        description += '''\tМужской\n
            Согласно социологическому исследованию, мужчины отдают предпочтение
            полуободковым металлическим оправам.\n'''
        s_vector[[29]] += 1.0 * emphasize_coef
        s_vector[[10]] += 0.5 * emphasize_coef
        s_vector[[11]] += 1.0 * emphasize_coef
        s_vector[[30]] += 1.0

    return s_vector, c_vector, description
