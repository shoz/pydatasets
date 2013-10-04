import numpy
import sys, copy, csv, re

class Datasets(dict):
    def __init__(self, path, id_key='ID', cleanup=[]):
        self.id_key = id_key
        self.load_csv(path, id_key, cleanup)
    def load_csv(self, path, id_key='ID', cleanup=[]):
        ret = {}
        with open(path) as f:
            reader = csv.reader(f)
            header = reader.next()
            for row in reader:
                d = dict(zip(header, row))
                id = self._cast_type(d[id_key])
                cleanup.append(id_key)
                d = self._ignore_keys(d, cleanup)
                self.__setitem__(id, self._cast_dict(d))
        return self
    def classify(self, key, ignore=[], append_id=False, with_label=True):
        classed = {}
        temp = copy.deepcopy(self)
        for k, v in sorted(temp.items()):
            klass = v[key]
            del v[key]
            if append_id:
                v[self.id_key] = k
            v = self._ignore_keys(v, ignore)
            if klass in classed:
                classed[klass].append(v)
            else:
                classed[klass] = [v]
        if with_label:
            return classed
        else:
            no_label = []
            for klass, data in sorted(classed.items()):
                l = []
                for e in data:
                    ll = []
                    for k, v in sorted(e.items()):
                        ll.append(v)
                    l.append(ll)
                no_label.append(l)
            return no_label
                
    def classify_with_average(self, key, ignore=[]):
        classed = self.classify(key, ignore)
        averaged = {}
        for klass, data in classed.items():
            d = {}
            for e in data:
                for k, v in e.items():
                    if k in d:
                        d[k] += v
                    else:
                        d[k] = v
            for k, v in d.items():
                d[k] = d[k] / float(len(data))
            averaged[klass] = d
        return averaged
    def _ignore_keys(self, d, ignore):
        temp = copy.deepcopy(d)
        for e in ignore:
            if e in temp:
                del temp[e]
        return temp
    def _cast_type(self, var):
        if re.match('\d+(\.\d+)', var):
            return float(var)
        elif re.match('\d+', var):
            return int(var)
        else:
            return str(var)
    def _cast_dict(self, d):
        for k, v in d.items():
            d[k] = self._cast_type(v)
        return d
    def labels(self):
        labels = []
        for id, data in self.items():
            for label, v in sorted(data.items()):
                labels.append(label)
            break
        return labels
    def col(self, key, ignore_ids=[]):
        col = []
        for id, data in sorted(self.items()):
            if not id in ignore_ids:
                col.append(data[key])
        return col
    def cols(self, ignore_labels=[], ignore_ids=[], with_label=True):
        cols = {}
        for label in self.labels():
            if not label in ignore_labels:
                cols[label] = self.col(label, ignore_ids=ignore_ids)
        if with_label:
            return cols
        else:
            return [v for k, v in sorted(cols.items())]
    def row(self, id, ignore_labels=[]):
        row = []
        for k, v in sorted(self[id].items()):
            if not k in ignore_labels:
                row.append(v)
        return row
    def rows(self, ignore_labels=[], ignore_ids=[], with_id=True):
        rows = {}
        for id, data in sorted(self.items()):
            if not id in ignore_ids:
                rows[id] = self.row(id, ignore_labels=ignore_labels)
        if with_id:
            return rows
        else:
            return [v for k, v in sorted(rows.items())]
    def row_std(self, id):
        return numpy.std(self.row(id))
    def col_std(self, key):
        return numpy.std(self.col(key))
    def row_mean(self, id):
        return numpy.average(self.row(id))
    def col_mean(self, key):
        return numpy.average(self.col(key))
    def col_means(self, with_label=True, ignore_labels=[]):
        means = {}
        for id, data in self.items():
            for k, v in sorted(data.items()):
                if not k in ignore_labels:
                    means[k] = self.col_mean(k)
            break
        if with_label:
            return means
        else:
            return [v for k, v in sorted(means.items())]
    def col_stds(self, with_label=True, ignore_labels=[]):
        stds = {}
        for id, data in self.items():
            for k, v in sorted(data.items()):
                if not k in ignore_labels:
                    stds[k] = self.col_std(k)
            break
        if with_label:
            return stds
        else:
            return [v for k, v in sorted(stds.items())]
    def normalize(self, ignore_labels=[]):
        stds = self.col_stds(ignore_labels=ignore_labels)
        means = self.col_means(ignore_labels=ignore_labels)
        for id, data in self.items():
            for k, v in data.items():
                if k in ignore_labels: continue
                self[id][k] = self._normalize(v, means[k], stds[k])
        return self
    def _normalize(self, value, mean, std):
        return (float(value) - float(mean)) / float(std)
    def normalize_with_other_sets(self, other, ignore_labels=[]):
        d1_cols = self.cols(ignore_labels=ignore_labels, with_label=True)
        d2_cols = other.cols(ignore_labels=ignore_labels, with_label=True)
        if len(d1_cols) != len(d2_cols):
            print 'Dimension Error'
        for key in d1_cols.keys():
            try:
                if key in d2_cols: pass
            except:
                print 'Key Error', key
        means = {}
        stds = {}
        for key in d1_cols.keys():
            joined = d1_cols[key] + d2_cols[key]
            means[key] = numpy.average(joined)
            stds[key] = numpy.std(joined)
        for id, data in self.items():
            for k, v in data.items():
                if k in ignore_labels: continue
                self[id][k] = self._normalize(v, means[k], stds[k])
        for id, data in other.items():
            for k, v in data.items():
                if k in ignore_labels: continue
                other[id][k] = self._normalize(v, means[k], stds[k])
        return self, other
