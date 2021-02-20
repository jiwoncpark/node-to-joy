from tqdm import tqdm
import multiprocessing

class SomeClass:
    def __init__(self, a):
        self.a = a
        self.b = 2.0

    def do_something(self, i):
        print(self.a[i]**self.b)
    def run(self):
        with multiprocessing.Pool(4) as pool:
            list(tqdm(pool.imap(self.do_something, range(len(self.a))),
                         total=len(self.a)))