import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import collections


class Vertex:
    """класс Вершина
    Так как решающие дерево состоит из вершин, то выделил под это отдельный класс
    """
    def __init__(self, my_data, left=None, right=None, index=None, threshold=None):
        """
        :param my_data: данные внутри вершины
        :param left: дочерняя левая вершина, те левый лист
        :param right: дочерняя правая вершина, те правый лист
        :param index: индекс фичи(стобца датафрейма)
        :param threshold: порог внутри фичи, который лучшим образо разделял выборку
        index и threshold нужно запомнить, что бы применять predict
        """
        self.my_data = my_data
        self.left = left
        self.right = right
        self.index = index
        self.threshold = threshold

    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1


class DecisionTree:
    index = 0

    def __init__(self, max_depth=None, min_samples=1, splitter='best', type_sample='max'):
        """
        :param max_depth: максимальная глубина дерева
        :param min_samples: ограничение на минималькое ко-во элементов в листе
        :param splitter best - полный перебор, повышает качество, но просели по времени.
        Все кроме best это варинт с выборочным перебором, что уменьшит время, но мы можем просесть по качеству немного.
        :param type_sample - отвечает за длину случайного подмножества из которого будет выбираться наилучший признак.
        len_sample принимает 2 варианта max и random
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.columns = None
        self.head_tree = None
        if splitter == 'best':
            self.splitter = self.splitter_best
        else:
            self.splitter = self.splitter_fast
        if type_sample == 'max':
            self.type_sample = self.sample_max
        elif type_sample == 'random':
            self.type_sample = self.sample_random
        else:
            raise ValueError('неизвестное значение для type_sample')

    def __repr__(self):
        return f"Tree deep:{self.max_depth}"

    @staticmethod
    def splitter_best(vertex, feat):
        return set(vertex[:, feat])

    @staticmethod
    def splitter_fast(vertex, feat):
        col = vertex[:, feat]
        std_ = col.std()
        std_ = 1 if std_ == 0 else std_
        step = int(len(col) / std_)
        step = 1 if step == 0 else step
        return np.unique(col)[::step]

    @staticmethod
    def sample_max(n_feats):
        return range(n_feats)

    @staticmethod
    def sample_random(n_feats):
        q = int(n_feats ** 0.5)
        arr = np.arange(n_feats)
        np.random.shuffle(arr)
        return arr[:q]

    @staticmethod
    def gini(rows: list) -> float:
        """
        Расчитывает критерий информативности gini
        :param rows: массив элементов, для которого мы рассчитывает gini
        """
        uniq_target = set(rows)
        lens_rows = len(rows)
        """
        sums = 0
        for my_target in uniq_target:
            doli = sum(rows == my_target) / lens_rows
            sums += doli * (1 - doli)
        return sums
        """
        return 1 - sum([(sum(rows == my_target) / lens_rows) ** 2 for my_target in uniq_target])

    def information(self, left: np.array, right: np.array) -> float:
        """
        :param left: левый лист
        :param right: правый лист
        :return: критерйи gini, информирует насколько хорошо мы нашли критерий для разделения вершины на листы
        """
        father = len(left) + len(right)
        return len(left) / father * self.gini(left[:, -1]) + len(right) / father * self.gini(right[:, -1])

    @staticmethod
    def partition(my_data: np.array, index: int, threshold: float) -> tuple:
        """
        Формирует правый и левый лист
        Мы делим нашу табличку(вершину) на 2 листа, путем того, что фильтруем строки таблицы.
        :param my_data: вершина, которую мы хотим разбить, формат: исходня таблица
        :param index: номер столбца из таблицы (фича)
        :param threshold: порог разбиения вершины на листы
        :return: левый и правый лист дерева
        """
        true_rows, false_rows = [], []
        """
        for row in my_data:
            if row[index] <= threshold:
                true_rows.append(row)
            else:
                false_rows.append(row)
        """
        [true_rows.append(row) if row[index] <= threshold else false_rows.append(row) for row in my_data]
        return np.array(true_rows), np.array(false_rows)

    def best_split(self, vertex: np.array) -> tuple:
        """
        вопрос, можно ли не перебирвть все значения столца, а просто взять максимальный?
        -Нет, нельяз, так как best_threshold и best_index параметры одной и тойже фичи,
        таким образом вершина разделится очень тупо, левый лист будет включать в себя весь датасет,
        а правый лист будет пустой

        Находим лучшие разбиение полным перебором, что по сути очень медленноо.
        В цикле я прохожусь по каждому стобцу(по каждой фиче) из моей таблички(вершины)
        и пытаюсь найти тот элемент фичи, который будет наилучем образом разделять мою вершину
        :param vertex: вершина, в которой будем искать значения для разбеения
        n_feats: кол-во столбцов
        :return: наилучшее значение фичи(порог для разделения), номер наилучшей фичи

        len(vertex[0]) - 1  так как таргет кранится в конце
        """
        n_feats = len(vertex[0]) - 1
        best_gini = float('inf')
        best_threshold = best_index = None
        for feat in self.type_sample(n_feats):
            index = feat
            values = self.splitter(vertex, feat)
            for threshold in values:
                true_rows, false_rows = self.partition(vertex, index, threshold)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gini = self.information(true_rows, false_rows)
                if gini < best_gini:
                    best_threshold, best_index, best_gini = threshold, index, gini

        return best_threshold, best_index

    def create_leaf(self, vertex: Vertex):
        """
        Добавляем правый и левый лист к нашей выршине
        :param vertex: вершина
        :return: None
        """
        vertex.threshold, vertex.index = self.best_split(vertex.my_data)
        left, right = self.partition(vertex.my_data, vertex.index, vertex.threshold)
        vertex.left = Vertex(left)
        vertex.right = Vertex(right)

    def fit(self, my_data, my_target=None):
        """рекурсивное обучение"""
        if isinstance(my_data, pd.DataFrame):
            self.columns = my_data.columns
            self.head_tree = tree = Vertex(np.column_stack([my_data.values, my_target]))
        elif isinstance(my_data, np.ndarray) and self.head_tree is None:
            if my_target is None:
                self.head_tree = tree = Vertex(my_data)
            else:
                self.head_tree = tree = Vertex(np.column_stack([my_data, my_target]))
        elif isinstance(my_data, Vertex):
            tree = my_data
        else:
            raise ValueError("не тот формат данных на входе")
        if len(tree.my_data) > self.min_samples and self.head_tree.depth() < self.max_depth \
        and len(tree.my_data) >= 2 and len(set(tree.my_data[:, -1])) > 1:
            self.create_leaf(tree)
            self.fit(tree.left)
            self.fit(tree.right)

    def leaf_target(self, np_data: np.array) -> list:
        """
        Собираем метки классов в итоговом листе
        :param np_data: фрейм для которого нужно предсказать результат
        :return: маисив ответов (меток класов) исходя из трейна
        """
        mas_target = []
        for row in np_data:
            tree = self.head_tree
            while tree.left or tree.right:
                if tree.right and row[tree.index] > tree.threshold:
                    tree = tree.right
                else:
                    tree = tree.left
            targets = []
            for target_ in tree.my_data:
                targets.append(target_[-1])
            mas_target.append(targets)
        return mas_target

    @staticmethod
    def count_target(mas_target: list) -> tuple:
        """
        Подсчитывает наиболее популярную метку метку класса
        :param mas_target:  массив из меток по классам
        :return: самый попоулярный класс, кол-во элементов самого популярного класса
        """
        dicts = collections.Counter(mas_target)
        return dicts.most_common(1)[0][0], dict(dicts)

    def predict(self, data_test):
        """
        предсказывает результат да данных
        :param data_test: dataframe
        :return: предсказания для данных
        """
        if isinstance(data_test, pd.DataFrame):
            test = data_test.values
        else:
            test = data_test
        mas_targets = self.leaf_target(test)
        for index in range(len(mas_targets)):
            mas_targets[index], _ = self.count_target(mas_targets[index])
        return mas_targets

    def predict_proba(self, data_test):
        """
        вероятностоное предсказание для данных
        :param data_test: dataframe
        :return: предсказания для данных
        """
        if isinstance(data_test, pd.DataFrame):
            test = data_test.values
        else:
            test = data_test
        mas_targets = self.leaf_target(test)
        for index in range(len(mas_targets)):
            _, dict_target = self.count_target(mas_targets[index])
            sums = sum(list(dict_target.values()))
            mas_target = []
            for key in dict_target:
                mas_target.append(dict_target[key] / sums)
            mas_targets[index] = mas_target
        return mas_targets


if __name__ == '__main__':
    """  к чему мы пришли? 
    Деррево работает и работает коректно.
    основыне различия с sklern:p
    1) параметр sk_tree.max_depth = my_tree.max_depth + 1
    те у меня параметр max_depth должен быть на 1 больше аналогичного значения max_depth в sk_tree.
    Мне кажется, что мой вариант более правильный, так что исправлять его не буду. 
    2) Раздиления просиходят различным способом, что означает splitter='best' я не нашел ни в фоициальной документации,
    ни на просторах инета. Но надо признать, что мой метод (полного перебра) работает лучше по качеству. 
    Но могу предположить, что дольше по времени.
    поискав немного еще нашел что sklern ссылаются на алгоритм на основе Фишера-Йейтса, это все что по сути говорилось,
    а этот алгос связан просто с случайной перестановкой, те не совсем то, что мне бы хотелось, скорее всего сильный 
    прирост по времени связан с использованием Cython 
    """

    np.random.RandomState(12)
    breast_cancer = load_breast_cancer()
    data, target = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names), breast_cancer.target
    # data['target'] = target
    # print(data)
    #
    data.columns = breast_cancer.feature_names

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=21)

    start_time = time.time()
    sk_tree = DecisionTreeClassifier(max_depth=1, random_state=21)
    sk_tree.fit(train_data, train_target)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    my_tree = DecisionTree(2, splitter='best')
    my_tree.fit(train_data, train_target)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    my_tree = DecisionTree(2, splitter='2')
    my_tree.fit(train_data, train_target)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(accuracy_score(test_target, sk_tree.predict(test_data)))
    print(accuracy_score(test_target, my_tree.predict(test_data)))