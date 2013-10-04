import unittest, copy, os
import numpy
from pydatasets.datasets import Datasets

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.path = os.path.dirname(__file__)
        self.sets = Datasets(self.path + '/data/testdata.csv')
        self.classed_sets = Datasets(self.path + '/data/testdata_classed.csv')
    def test_load_csv(self):
        sets = Datasets(self.path + '/data/testdata.csv')
        for k, v in sets.items():
            assert type(k) == int
            assert type(v) == dict
        assert sets[1] == {'D1':1, 'D2':2, 'D3':3}
        assert sets[2] == {'D1':4, 'D2':5, 'D3':6}
        assert sets[3] == {'D1':7, 'D2':8, 'D3':9}
        sets = Datasets(self.path + '/data/testdata.csv', cleanup=['D2'])
        assert sets[1] == {'D1':1, 'D3':3}
        assert sets[2] == {'D1':4, 'D3':6}
        assert sets[3] == {'D1':7, 'D3':9}
    def test_cast_type(self):
        sets = self.sets
        assert sets._cast_type('1') == int('1')
        assert sets._cast_type('1.0') == float('1.0')
        assert sets._cast_type('test') == str('test')
    def test_get_row(self):
        sets = self.sets
        assert sets.row(1) == [1, 2, 3]
        assert sets.row(2) == [4, 5, 6]
        assert sets.row(3) == [7, 8, 9]
        assert sets.row(1, ignore_labels=['D2']) == [1, 3]
        assert sets.row(2, ignore_labels=['D2']) == [4, 6]
        assert sets.row(3, ignore_labels=['D2']) == [7, 9]
    def test_get_col(self):
        sets = self.sets
        assert sets.col('D1') == [1, 4, 7]
        assert sets.col('D2') == [2, 5, 8]
        assert sets.col('D3') == [3, 6, 9]
        assert sets.col('D1', ignore_ids=[1]) == [4, 7]
        assert sets.col('D2', ignore_ids=[1]) == [5, 8]
        assert sets.col('D3', ignore_ids=[1]) == [6, 9]
    def test_get_rows(self):
        sets = self.sets
        assert sets.rows(with_id=False) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert sets.rows(ignore_ids=[1],
                         with_id=False) == [[4, 5, 6], [7, 8, 9]]
        assert sets.rows(ignore_labels=['D1'],
                         with_id=False) == [[2, 3], [5, 6], [8, 9]]
        assert sets.rows(ignore_ids=[1],
                         ignore_labels=['D1'],
                         with_id=False) == [[5, 6], [8, 9]]
        assert sets.rows(with_id=True) == {1: [1, 2, 3],
                                           2: [4, 5, 6],
                                           3: [7, 8, 9]}
    def test_get_cols(self):
        sets = self.sets
        assert sets.cols(with_label=False) == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        assert sets.cols(ignore_ids=[2],
                         with_label=False) == [[1, 7], [2, 8], [3, 9]]
        assert sets.cols(ignore_labels=['D1'],
                         with_label=False) == [[2, 5, 8], [3, 6, 9]]
        assert sets.cols(ignore_ids=[1],
                         ignore_labels=['D1'],
                         with_label=False) == [[5, 8], [6, 9]]
        assert sets.cols(with_label=True) == {'D1':[1, 4, 7],
                                              'D2':[2, 5, 8],
                                              'D3':[3, 6, 9]}
    def test_get_labels(self):
        sets = self.sets
        assert sets.labels() == ['D1', 'D2', 'D3']
    def test_getitem(self):
        sets = self.sets
        assert self.sets[1] == {'D1': 1, 'D2': 2, 'D3': 3}
    def test_classify(self):
        sets = self.classed_sets
        classed = sets.classify('Class', append_id=True)
        print classed
        assert classed[1] == [{'ID':1, 'D1':1, 'D2':2, 'D3':3},
                              {'ID':4, 'D1':0, 'D2':1, 'D3':2}], classed[1]
        assert classed[2] == [{'ID':2, 'D1':4, 'D2':5, 'D3':6},
                              {'ID':5, 'D1':3, 'D2':4, 'D3':5}], classed[2]
        assert classed[3] == [{'ID':3, 'D1':7, 'D2':8, 'D3':9},
                              {'ID':6, 'D1':6, 'D2':7, 'D3':8}], classed[3]
        classed = sets.classify('Class')
        assert classed[1] == [{'D1':1, 'D2':2, 'D3':3},
                              {'D1':0, 'D2':1, 'D3':2}], classed[1]
        assert classed[2] == [{'D1':4, 'D2':5, 'D3':6},
                              {'D1':3, 'D2':4, 'D3':5}], classed[2]
        assert classed[3] == [{'D1':7, 'D2':8, 'D3':9},
                              {'D1':6, 'D2':7, 'D3':8}], classed[3]
        classed = sets.classify('Class', with_label=False)
        assert classed == [ [[1, 2, 3], [0, 1, 2]],
                            [[4, 5, 6], [3, 4, 5]],
                            [[7, 8, 9], [6, 7, 8]] ]
    def test_classify_with_average(self):
        sets = self.classed_sets
        classed = sets.classify_with_average('Class', ignore=['ID'])
        assert classed[1] == {'D1': 0.5, 'D2': 1.5, 'D3': 2.5}, classed[1]
        assert classed[2] == {'D1': 3.5, 'D2': 4.5, 'D3': 5.5}, classed[2]
        assert classed[3] == {'D1': 6.5, 'D2': 7.5, 'D3': 8.5}, classed[3]
    def test_normalize(self):
        sets = copy.deepcopy(self.sets)
        sets.normalize()
        for col in sets.cols(with_label=False):
            assert numpy.average(col) == 0, numpy.average(col)
            assert numpy.std(col) == 1, numpy.std(col)
        sets = copy.deepcopy(self.sets)
        sets.normalize(ignore_labels='D1')
        cols = sets.cols(with_label=False)
        assert cols[0][0] == 1
        assert cols[0][1] == 4
        assert cols[0][2] == 7
        for col in sets.cols(with_label=False)[1:]:
            assert numpy.average(col) == 0, numpy.average(col)
            assert numpy.std(col) == 1, numpy.std(col)
    def test_normalize_with_other_sets(self):
        s1 = self.sets
        s2 = self.classed_sets
        s1.normalize_with_other_sets(s2, ignore_labels=['ID', 'Class'])
        for label in s1.labels():
            joined = s1.col(label) + s2.col(label)
            average = numpy.average(joined)
            assert average > -0.00000001 and average < 0.00000001
            assert numpy.std(joined) == 1
    def test_row_mean(self):
        pass
    def test_col_mean(self):
        pass
    def test_row_std(self):
        pass
    def test_col_std(self):
        pass
    def test_col_means(self):
        sets = self.sets
        means = sets.col_means()
        assert means == {'D1': 4, 'D2': 5, 'D3':6}, means
        means = sets.col_means(with_label=False)
        assert means == [4, 5, 6], means
    def test_col_stds(self):
        sets = self.sets
        stds = sets.col_stds()
        assert stds == {'D1': 2.4494897427831779,
                        'D2': 2.4494897427831779,
                        'D3': 2.4494897427831779}, stds
        stds = sets.col_stds(with_label=False)
        assert stds == [2.4494897427831779,
                        2.4494897427831779,
                        2.4494897427831779], stds

if __name__ == '__main__':
    unittest.main()
