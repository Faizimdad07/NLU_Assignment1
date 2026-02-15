import os
import re
import math
import random
from collections import Counter, defaultdict

DATASET_DIR = 'dataset'

STOPWORDS = set([
    'the','a','an','and','or','is','are','was','were','in','on','at','of','for','to','with','by','that','this','it','from','as','be'
])

def list_files(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.txt')]

def load_docs(folder):
    docs = []
    for fp in sorted(list_files(folder)):
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            docs.append(f.read().strip())
    return docs

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    toks = [t for t in text.split() if t and t not in STOPWORDS]
    return toks

def build_vocab(docs, min_freq=1):
    c = Counter()
    for d in docs:
        c.update(d)
    vocab = {w for w,f in c.items() if f>=min_freq}
    return sorted(vocab)

def doc_to_bow(doc, vocab_index):
    ctr = Counter(doc)
    vec = {}
    for w,i in vocab_index.items():
        import os  # os for filesystem operations
        import math  # math for logs and sqrt
        import re  # regex for preprocessing
        from collections import defaultdict, Counter  # collections utilities

        def preprocess(text):  # lowercase and remove non-alphanum
            text = text.lower()  # lowercase
            text = re.sub(r"[^a-z0-9\s]"," ", text)  # remove punctuation
            text = re.sub(r"\s+"," ", text).strip()  # normalize whitespace
            return text  # return cleaned text

        def build_vocab(docs, min_df=1):  # build vocab from docs with min document freq
            df = defaultdict(int)  # document frequency
            for d in docs:  # for each doc
                seen = set()  # track seen tokens
                for w in d.split():  # for each token
                    if w in seen: continue  # only count once per doc
                    df[w] += 1  # increment df
                    seen.add(w)  # mark seen
            vocab = {w:i for i,(w,c) in enumerate(df.items()) if c>=min_df}  # index tokens
            return vocab  # return mapping

        def doc_to_bow(doc, vocab):  # convert doc to bag-of-words vector
            vec = [0]*len(vocab)  # zero vector
            for w in doc.split():  # count tokens
                if w in vocab:
                    vec[vocab[w]] += 1  # increment
            return vec  # return vector

        def build_tfidf(docs, vocab):  # compute tf-idf vectors
            N = len(docs)  # number docs
            df = [0]*len(vocab)  # doc freq per term
            for d in docs:  # compute df
                seen = set()
                for w in d.split():
                    if w in vocab and w not in seen:
                        df[vocab[w]] += 1
                        seen.add(w)
            idf = [math.log((N+1)/(1+dfi))+1 for dfi in df]  # smoothed idf
            tfidf_docs = []  # output list
            for d in docs:  # for each doc
                vec = [0]*len(vocab)  # term freq vector
                for w in d.split():
                    if w in vocab:
                        vec[vocab[w]] += 1
                for i in range(len(vec)):  # apply idf
                    vec[i] = vec[i]*idf[i]
                tfidf_docs.append(vec)  # append
            return tfidf_docs  # return list of vectors

        def get_ngrams(text, n=2):  # generate n-grams joined by underscore
            toks = text.split()  # tokens
            ng = []  # ngram list
            for i in range(len(toks)-n+1):
                ng.append('_'.join(toks[i:i+n]))  # join
            return ng  # return ngrams

        class MultinomialNB:  # simple multinomial Naive Bayes
            def __init__(self):
                self.class_log_prior = {}  # log priors
                self.feature_log_prob = {}  # log prob per feature
                self.vocab = None  # vocab set

            def fit(self, docs, labels):  # fit model from tokenized docs
                n_docs = len(docs)  # number docs
                classes = set(labels)  # class set
                self.vocab = set()  # build vocab set
                for d in docs:
                    for w in d.split():
                        self.vocab.add(w)
                V = len(self.vocab)  # vocab size
                counts = {c:defaultdict(int) for c in classes}  # per-class counts
                doc_count = {c:0 for c in classes}  # doc counts
                total_tokens = {c:0 for c in classes}  # token totals
                for d,l in zip(docs, labels):  # accumulate
                    doc_count[l] += 1
                    for w in d.split():
                        counts[l][w] += 1
                        total_tokens[l] += 1
                for c in classes:  # compute log priors and feature log probs
                    self.class_log_prior[c] = math.log(doc_count[c]+1) - math.log(n_docs + len(classes))
                    self.feature_log_prob[c] = {}
                    for w in self.vocab:
                        self.feature_log_prob[c][w] = math.log(counts[c].get(w,0)+1) - math.log(total_tokens[c] + V)

            def predict(self, doc):  # predict class for a single doc
                best = None  # best label
                best_score = None  # best score
                for c in self.class_log_prior:  # score each class
                    s = self.class_log_prior[c]
                    for w in doc.split():
                        if w in self.feature_log_prob[c]:
                            s += self.feature_log_prob[c][w]
                    if best is None or s>best_score:
                        best = c
                        best_score = s
                return best  # return best class

        def knn_predict(train_docs, train_labels, test_doc, k=3):  # k-NN with cosine similarity
            def norm(v):
                return math.sqrt(sum(x*x for x in v))  # vector norm
            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))  # dot product
            test_vec = test_doc  # test vector
            sims = []  # similarities
            for d,l in zip(train_docs, train_labels):  # compute similarity to each train
                s = dot(d, test_vec)/(norm(d)+1e-12)/(norm(test_vec)+1e-12)  # cosine
                sims.append((s,l))  # append
            sims.sort(reverse=True)  # sort by similarity
            top = sims[:k]  # top k
            votes = defaultdict(int)  # vote counts
            for s,l in top:
                votes[l] += 1  # vote
            return max(votes, key=votes.get)  # return most voted label

        class SGDLogistic:  # simple SGD logistic regression
            def __init__(self, lr=0.1, epochs=5):
                self.lr = lr  # learning rate
                self.epochs = epochs  # epochs
                self.w = None  # weights
                self.b = 0.0  # bias

            def fit(self, X, y):  # fit weights on dense vectors X
                n = len(X)  # n samples
                m = len(X[0])  # feature dim
                self.w = [0.0]*m  # init weights
                for _ in range(self.epochs):  # epochs
                    for xi, yi in zip(X,y):
                        z = sum(a*b for a,b in zip(self.w, xi)) + self.b  # linear score
                        pred = 1.0/(1.0+math.exp(-z))  # sigmoid
                        g = pred - (1 if yi==1 else 0)  # gradient
                        for i in range(m):
                            self.w[i] -= self.lr * g * xi[i]  # update weight
                        self.b -= self.lr * g  # update bias

            def predict(self, X):  # predict binary labels for list X
                out = []  # outputs
                for xi in X:
                    z = sum(a*b for a,b in zip(self.w, xi)) + self.b  # score
                    out.append(1 if z>0 else 0)  # threshold
                return out  # return list

        def run():  # main experiment run
            train_texts = []  # train texts
            train_labels = []  # train labels
            test_texts = []  # test texts
            test_labels = []  # test labels
            for split in ['train','test']:  # iterate splits
                for label in ['politics','sports']:  # iterate classes
                    folder = os.path.join('dataset', split, label)  # folder path
                    if not os.path.exists(folder):
                        continue  # skip if missing
                    for fname in os.listdir(folder):  # for each file
                        path = os.path.join(folder, fname)  # file path
                        with open(path,'r',encoding='utf-8') as f:  # read file
                            txt = f.read()  # read content
                        txt = preprocess(txt)  # preprocess
                        if split=='train':
                            train_texts.append(txt)  # add to train
                            train_labels.append(label)  # add label
                        else:
                            test_texts.append(txt)  # add to test
                            test_labels.append(label)  # add label

            vocab = build_vocab(train_texts, min_df=2)  # build vocab
            X_train = [doc_to_bow(d, vocab) for d in train_texts]  # bow train
            X_test = [doc_to_bow(d, vocab) for d in test_texts]  # bow test

            nb = MultinomialNB()  # NB classifier
            nb.fit(train_texts, train_labels)  # fit NB on tokenized texts
            preds_nb = [nb.predict(d) for d in test_texts]  # NB predictions

            preds_knn = []  # kNN predictions
            for d in X_test:  # for each test bow
                p = knn_predict(X_train, train_labels, d, k=5)  # predict
                preds_knn.append(p)  # append

            tfidf_train = build_tfidf(train_texts, vocab)  # tfidf train
            tfidf_test = build_tfidf(test_texts, vocab)  # tfidf test
            y_train = [1 if l=='sports' else 0 for l in train_labels]  # binary labels
            y_test = [1 if l=='sports' else 0 for l in test_labels]  # binary labels
            sgd = SGDLogistic(lr=0.01, epochs=3)  # SGD logistic
            sgd.fit(tfidf_train, y_train)  # train
            preds_sgd = sgd.predict(tfidf_test)  # predict binary
            preds_sgd = ['sports' if p==1 else 'politics' for p in preds_sgd]  # convert labels

            def acc(a,b):  # accuracy helper
                c = sum(1 for x,y in zip(a,b) if x==y)  # correct count
                return 100.0 * c / len(a) if a else 0.0  # percentage

            results = {  # collect results
                'NB': acc(preds_nb, test_labels),
                'kNN': acc(preds_knn, test_labels),
                'SGD': acc(preds_sgd, test_labels)
            }

            with open('results_prob4.txt','w',encoding='utf-8') as f:  # write results
                for k,v in results.items():
                    f.write(f"{k}: {v:.2f}\n")  # write each
            print('Results written to results_prob4.txt')  # notify

            with open('B23CM1016_report.txt','w',encoding='utf-8') as f:  # write report
                f.write('Problem 4 Report\n')  # header
                f.write('==================\n')  # underline
                f.write('Techniques: MultinomialNB, k-NN (cosine), TF-IDF + SGD Logistic\n')  # techniques
                f.write('\nResults:\n')  # results heading
                for k,v in results.items():
                    f.write(f"{k}: {v:.2f}%\n")  # results lines

        if __name__=='__main__':
            run()  # execute run
