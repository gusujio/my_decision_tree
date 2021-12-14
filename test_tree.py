import pandas as pd

from my_DecisionTree import Vertex, DecisionTree
import numpy as np


def test_vertex_depth():
    v2_1 = Vertex([4, 5, 6])
    v2_2 = Vertex([7, 8, 9])
    v1 = Vertex([1, 2, 3], v2_1, v2_2)
    assert v1.depth() == 2
    v3_1_1 = Vertex([4, 5, 6])
    v3_1_2 = Vertex([4, 5, 6])
    v2_2 = Vertex([7, 8, 9], v3_1_1, v3_1_2)
    v1 = Vertex([1, 2, 3], v2_1, v2_2)
    assert v1.depth() == 3


def test_gini():
    """
    проверил работоспособность gini и information. Если верить этой статье. К методам sklern так и не подобрался
    https://www.analyticsvidhya.com/blog/2021/03/how-to-select-best-split-in-decision-trees-gini-impurity/
    трансформация нужна для коректной работы внутри дерева
    """
    tree = DecisionTree(2, 1)
    left = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]).T
    right = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    assert round(tree.information(left, right), 3) == 0.32
    left = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]).T
    right = np.array([[1, 1, 0, 0, 0, 0]]).T
    assert round(tree.information(left, right), 3) == 0.476


def test_best_split():
    """находим то значение в колонке, которе наилучшим образ делит нашу выборку"""
    tree = DecisionTree(2, 1)
    vertex = np.array([[1, 2, 3, 1],
                       [4, 5, 6, 1],
                       [11, 2, 3, 0],
                       [10, 4, 5, 0]])
    assert tree.best_split(vertex) == (4, 0)

    vertex = np.array([[1, 2, 3, 1],
                      [4, 5, 2, 1],
                      [1, 2, 10, 0],
                      [4, 4, 5, 0]])
    assert tree.best_split(vertex) == (3, 2)

    vertex = np.array([[1, 2, 7, 1],
                      [4, 10, 6, 1],
                      [1, 2, 3, 0],
                      [4, 10, 5, 0]])
    assert tree.best_split(vertex) == (5, 2)


def test_create_leaf():
    """
     проверил способность переобучаться, какждая строка способна попасть в отдельный лист данных, что радует
     все методы стоязие выше совместно работают коректно
    """
    vertex = np.array([[1, 2, 3, 1],
                       [4, 5, 6, 1],
                       [11, 2, 3, 0],
                       [10, 4, 5, 0]])
    vertex = Vertex(vertex)
    tree = DecisionTree(2, 1)
    tree.create_leaf(vertex)
    tree.create_leaf(vertex.left)
    assert np.all(vertex.left.left.my_data == [[1, 2, 3, 1]])
    assert np.all(vertex.right.my_data == [[11, 2, 3, 0], [10, 4, 5, 0]])


def test_fit():
    """
    Протестил вариант с полным переобучением, вариант с max_depth и вариант с min_samples.
    """
    my_tree = DecisionTree(4)
    data = np.array([[1, 2, 3, 1],
                       [4, 5, 6, 1],
                       [11, 2, 3, 0],
                       [10, 4, 5, 0]])
    my_tree.fit(data)
    assert np.all(my_tree.head_tree.left.left.my_data == [[1, 2, 3, 1]])
    assert np.all(my_tree.head_tree.left.right.my_data == [[4, 5, 6, 1]])
    assert np.all(my_tree.head_tree.right.left.my_data == [[10, 4, 5, 0]])
    assert np.all(my_tree.head_tree.right.right.my_data == [[11, 2, 3, 0]])

    my_tree = DecisionTree(2)
    my_tree.fit(data)
    assert np.all(my_tree.head_tree.left.my_data == [[1, 2, 3, 1], [4, 5, 6, 1]])
    assert np.all(my_tree.head_tree.right.my_data == [[11, 2, 3, 0], [10, 4, 5, 0]])

    my_tree = DecisionTree(2, min_samples=4)
    my_tree.fit(data)
    assert my_tree.head_tree.left is None and  my_tree.head_tree.right is None


def test_leaf_target():
    """для каждого предсказания собираем массив из всех меток, которые по его пути в листе"""
    my_tree = DecisionTree(4)
    data = np.array([[1, 2, 3, 1],
                       [4, 5, 6, 1],
                       [11, 2, 3, 0],
                       [10, 4, 5, 0]])
    my_tree.fit(data)
    pred = my_tree.leaf_target(data[:, :-1])
    assert [[1], [1], [0], [0]] == pred

    my_tree = DecisionTree(2)
    my_tree.fit(data)
    pred = my_tree.leaf_target(data[:, :-1])
    assert [[1, 1], [1, 1], [0, 0], [0, 0]] == pred


def test_count_target():
    my_tree = DecisionTree()
    assert my_tree.count_target([1, 1, 2]) == (1, {1: 2, 2: 1})
    assert my_tree.count_target([1, 1, 2, 2, 3, 3, 3]) == (3, {1: 2, 2: 2, 3: 3})
    assert my_tree.count_target([1, 0, 0]) == (0, {1: 1, 0: 2})


def test_predict():
    my_tree = DecisionTree(2)
    data = np.array([[1, 2, 3, 1],
                     [4, 5, 6, 1],
                     [12, 5, 6, 1],
                     [1, 2, 3, 0],
                     [11, 2, 3, 0],
                     [10, 4, 5, 0]])
    my_tree.fit(data)
    assert np.all(my_tree.predict(data) == [0, 1, 1, 0, 0, 0])
    my_tree = DecisionTree(3)
    my_tree.fit(data)
    assert np.all(my_tree.predict(data) == [1, 1, 1, 1, 0, 0])


def test_predict_proba():
    my_tree = DecisionTree(2)
    data = np.array([[1, 2, 3, 1],
                     [4, 5, 6, 1],
                     [12, 5, 6, 1],
                     [1, 2, 3, 0],
                     [11, 2, 3, 0],
                     [10, 4, 5, 0]])
    my_tree.fit(data)
    assert np.all(my_tree.predict_proba(data) == [[0.25, 0.75], [1.0], [1.0], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]])
    my_tree = DecisionTree(3)
    my_tree.fit(data)
    assert np.all(my_tree.predict_proba(data) == [[0.5, 0.5], [1.0], [1.0], [0.5, 0.5], [1.0], [1.0]])

