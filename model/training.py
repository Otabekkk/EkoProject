# import os
# import shutil
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import pickle

# # Путь к TrashNet
# data_dir = r'C:\Users\TM\Desktop\Project\model\TrashNet'

# # Пути для train, validation, test
# train_dir = 'C:\\Users\\TM\\Desktop\\Project\\model\\train'
# validation_dir = 'C:\\Users\\TM\\Desktop\\Project\\model\\validation'
# test_dir = 'C:\\Users\\TM\\Desktop\\Project\\model\\test'

# # Создаем папки, если их нет
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(validation_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Классы
# classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# # Разделение данных
# for cls in classes:
#     # Создаем папки для каждого класса
#     os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
#     os.makedirs(os.path.join(validation_dir, cls), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, cls), exist_ok=True)


#     files = os.listdir(os.path.join(data_dir, cls))
#     train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
#     val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)


#     for f in train_files:
#         shutil.copy(os.path.join(data_dir, cls, f), os.path.join(train_dir, cls, f))
#     for f in val_files:
#         shutil.copy(os.path.join(data_dir, cls, f), os.path.join(validation_dir, cls, f))
#     for f in test_files:
#         shutil.copy(os.path.join(data_dir, cls, f), os.path.join(test_dir, cls, f))


# img_width, img_height = 150, 150
# batch_size = 32

# train_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')


# with open('class_labels.pkl', 'wb') as f:
#     pickle.dump(train_generator.class_indices, f)


# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(train_generator.num_classes, activation='softmax'))


# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# epochs = 10
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size)


# test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
# print(f'Test accuracy: {test_acc}')

# model.save('waste_classification_model.h5')

# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pickle
# import os

# # Пути к данным
# train_dir = 'C:\\Users\\TM\\Desktop\\Project\\model\\train'
# validation_dir = 'C:\\Users\\TM\\Desktop\\Project\\model\\validation'
# test_dir = 'C:\\Users\\TM\\Desktop\\Project\\model\\test'

# # Размер изображения и batch_size
# IMG_SIZE = 224
# BATCH_SIZE = 32

# # Аугментация данных
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Генераторы данных
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # Сохраняем классы
# with open('class_labels.pkl', 'wb') as f:
#     pickle.dump(train_generator.class_indices, f)

# # Загружаем предобученную MobileNetV2 без верхних слоев (include_top=False)
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# # Замораживаем слои базовой модели
# base_model.trainable = False

# # Создаем новую модель
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(len(train_generator.class_indices), activation='softmax')  # Количество классов
# ])

# # Компиляция модели
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Обучение модели
# epochs = 10
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE
# )

# # Оценка точности на тестовых данных
# test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
# print(f'Test accuracy: {test_acc}')

# # Сохраняем модель
# model.save('waste_classification_transfer.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle


model = load_model('waste_classification_transfer.h5')

with open('class_labels.pkl', 'rb') as f:
    class_indices = pickle.load(f)

class_labels = list(class_indices.keys())


def get_recycling_recommendations(waste_type):
    recommendations = {
        'glass': '''Стекло — это один из самых легко перерабатываемых материалов. 
        Его можно перерабатывать бесконечно, не теряя качества. Для утилизации стеклянных отходов:
        - Помойте стеклянные бутылки и банки от остатков пищи и напитков.
        - Убедитесь, что стекло не содержит загрязняющих веществ, таких как пластик или бумага.
        - Не смешивайте стекло с другими типами отходов.
        - Стекло, как правило, перерабатывается в новые бутылки, банки или другие стеклянные изделия.
        - Переработка стекла помогает уменьшить объем отходов на свалках и сокращает потребность в добыче песка для производства нового стекла.''',

        'plastic': '''Пластик — это материал, который может быть переработан, но требует сортировки по типам. 
        Важно:
        - Разделяйте пластик по типам: PET, HDPE, PVC и другие. Это поможет улучшить качество переработки.
        - Убедитесь, что пластиковая тара очищена от остатков пищи и напитков.
        - Некоторые виды пластиковых бутылок и упаковок можно перерабатывать в новые пластиковые изделия, такие как одежда, мебель и строительные материалы.
        - Использование переработанного пластика помогает сократить количество отходов на свалках и снизить потребность в производстве нового пластика.''',

        'paper': '''Бумага и картон легко поддаются переработке и могут быть превращены в новые бумажные изделия.
        Однако для переработки важно:
        - Очистить бумагу от загрязнений: не выбрасывайте бумагу с остатками пищи или масла (например, пиццы).
        - Разделяйте картон от обычной бумаги, так как для их переработки требуются разные методы.
        - Переработанная бумага используется для производства туалетной бумаги, коробок, газет и даже в качестве сырья для новых упаковок.
        - При переработке бумаги уменьшается вырубка лесов, и снижается потребность в древесном сырье.''',

        'metal': '''Металлические отходы, такие как алюминиевые банки, железные и стальные изделия, можно перерабатывать бесконечно.
        Рекомендации:
        - Убедитесь, что металлические отходы очищены от пищи и других загрязнений.
        - Металлы, такие как алюминий, можно перерабатывать в новые банки, фольгу, а также в строительные материалы.
        - Железо и сталь перерабатываются в новые изделия, такие как бытовая техника, автомобили, а также строительные конструкции.
        - Переработка металла помогает сократить потребность в добыче новых руд и снижает загрязнение окружающей среды.''',

        'organic': '''Органические отходы (например, остатки пищи и садовые отходы) могут быть переработаны в компост.
        Вот как правильно утилизировать органические отходы:
        - Собирайте пищевые отходы, такие как овощные и фруктовые остатки, кофейные зерна, яичные скорлупки.
        - Садовые отходы, такие как трава, листья, ветки, также могут быть переработаны в компост.
        - Компостирование помогает уменьшить объем отходов, а полученный компост можно использовать как удобрение для садов и огородов.
        - Избегайте добавления в компост мясных продуктов и молочных продуктов, так как они могут привлечь вредителей.''',

        'hazardous': '''Опасные отходы, такие как батарейки, химикаты, лекарства и некоторые виды электроприборов, требуют особого подхода.
        Рекомендации:
        - Для батареек и аккумуляторов: сдавайте их в специализированные пункты сбора. Никогда не выбрасывайте их в обычный мусор.
        - Химические вещества и лекарства можно сдавать в аптечные пункты или специальные центры переработки.
        - Электронные устройства, такие как старые телефоны, телевизоры, холодильники, должны сдавать в пункты сбора электроотходов.
        - Опасные отходы не должны попадать на свалки, так как они могут загрязнять воду и почву.''',

        'textiles': '''Текстильные отходы, такие как старая одежда и ткань, могут быть переработаны и использованы повторно.
        Как утилизировать текстиль:
        - Раздайте или сдайте ненужную одежду в благотворительные организации или на переработку.
        - Старые ткани могут быть использованы для изготовления новых текстильных изделий, таких как мебельные ткани или изоляционные материалы.
        - Обратите внимание на тканевые изделия, содержащие синтетические материалы, так как их переработка может быть сложнее.
        - Не выбрасывайте одежду с химическим загрязнением (например, одежду, обработанную красками или химическими веществами).'''

    }
    return recommendations.get(waste_type, 'Для этого типа отходов рекомендации не найдены.')


def classify_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    waste_type = class_labels[predicted_class[0]]
    recommendations = get_recycling_recommendations(waste_type)

    return waste_type, recommendations