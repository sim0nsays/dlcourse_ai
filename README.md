This repo relates to Feb-May 2019 session of the Deep Learning course (https://dlcourse.ai)

# Course roadmap

## ЛЕКЦИИ И ЗАДАНИЯ
### Лекция 1: 
Введение	20 февраля, 
8:40 Мск	О чем курс, что такое machine learning и deep learning, основные домены - computer vision, NLP, speech recognition, reinforcement learning.

### Семинар 1: 
Python, numpy, notebooks		Краткий обзор инструментария, необходиомого для курса - Python, Jupyter, numpy. Google Colab как среда выполнения Jupyter Notebooks в облаке.

### Лекция 2: 
Элементы машинного обучения	27 февраля, 
8:40 Мск	Обзор задачи supervised learning. K-nearest neighbor как пример простого алгоритма обучения. Тренировочная и тестовые выборки. Гиперпараметры, их подбор с помощью validation set и cross-validation. Общая последовательность действий при тренировке и валидации моделей (Machine Learning Flow).

### Семинар 2: 
Установка окружения для заданий		Установка окружения, необходимого для решения заданий. Некоторые детали KNN.

### Задание 1, Часть 1: 
K-nearest neighbor		Знакомство с Python и numpy, реализация K-nearest neighbor classifier руками. Выбор гиперпараметра с помощью cross-validation.

### Лекция 3: 
Нейронные сети	
6 марта, 8:40 Мск	
Линейный классификатор - нейронная сеть с одним слоем. Softmax, функция потерь cross-entropy. Тренировка с помощью стохастического градиентного спуска, регуляризация весов. Многослойные нейронные сети, fully-connected layers. Алгоритм backpropagation.

### Семинар 3: 
Вычисление градиентов		Детальный разбор вычисления градиентов softmax и cross-entropy.

### Задание 1, Часть 2: 
Линейный классификатор		Реализация линейного классификатора, подсчет градиентов и тренировка с помощью SGD своими руками.	Задание

### Лекция 4: 
PyTorch и подробности	
13 марта, 8:40 Мск	
Backpropagation с матрицами. Введение в PyTorch. Инициализация весов. Улучшенные алгоритмы градиентного спуска (Adam, RMSProp, итд).

### Задание 2, Часть 1: 
Нейронные сети		Реализация своей собственной многослойной нейронной сети и ее тренировки.	Задание

### Лекция 5: 
Нейронные сети на практике	
20 марта, 8:40 Мск	
GPUs. Процесс тренировки и overfitting/underfitting на практике,. Learning rate annealing. Batch Normalization. Ансамбли. Что нового в 2018.

### Задание 2, Часть 2: 
PyTorch		Реализация нейросети на PyTorch, практика тренировки и визуализации предсказаний модели.	Задание

### Лекция 6: 
Convolutional Neural Networks	
27 марта, 8:40 Мск	
Convolution и pooling layers. Эволюция архитектур: LeNet, AlexNet, VGG, ResNet. Transfer learning. Аугментации.	

### Задание 3: 
Convolutional Neural Networks		
Реализация Convolutional Neural Networks руками и на PyTorch

### Лекция 7: 
Segmentation и Object Detection 
(Владимир Игловиков)	
3 апреля, 8:40 Мск	
Более сложные задачи компьютерного зрения - сегментация (segmentation) и нахождение объектов на изображении (object detection).	

### Задание 4: 
Hotdog or Not		
Использование методов transfer learning и fine tuning на примере распознавания хотдогов.

### Лекция 8: 
Metric Learning, Autoencoders, GANs	
10 апреля, 8:40 Мск	Metric Learning на примере распознавания лиц, обзор некоторых методов unsupervised learning в DL

### Лекция 9: 
Введение в NLP, word2vec	
17 апреля, 8:40 Мск	
Краткий обзор области обработки естественного языка и применения deep learning к ней на примере word2vec.

### Задание 5: 
Word2Vec		Реализация word2vec на PyTorch на маленьком наборе данных.	Задание

### Лекция 10: 
Recurrent Neural Networks	
24 апреля, 8:40 Мск	
Применение рекуррентных нейронных сетей (recurrent neural networks) в задачах распознавания естественного языка. Детали архитектуры LSTM.

### Задание 6: 
RNNs		Использование LSTM для определения части речи (Part of Speech Tagging). Адаптировано из курса Даниила Анастасьева с разрешения автора.
Задание

### Лекция 11: 
Аудио и распознавание речи
(Юрий Бабуров)	
1 мая, 8:40 Мск	
Применение методов deep learning к задаче распознавания речи. Краткий обзор других задач, связанных с аудио.

### Лекция 12: 
Attention	
8 мая, 8:40 Мск	
Использование механизма Attention в NLP на примере задачи машинного перевода. Архитектура Transformer, современное развитие.
Задание: 
Написать пост о статье		Прочитайте и опишите в посте одну из современных статей в области deep learning!	Инструкции

### Лекция 13: 
Reinforcement Learning	
15 мая, 8:40 Мск	
Введение в обучение с подкреплением (reinforcement learning), использование методов deep learning. Базовые алгоритмы - Policy Gradients и Q-Learning

### Лекция 14: 
Еще о Reinforcement Learning	
22 мая, 8:40 Мск	
Model-based RL на примере AlphaZero. Критика и некоторые возможные пути развития области.	



