from multiprocessing import Process, Lock, Pipe
from multiprocessing.connection import wait


def f(w, l, i):
    l.acquire()
    try:
        w.send(('hello',i))
        # print('hello world', i)
    finally:
        l.release()
        pass
    w.close()
    

if __name__ == '__main__':
    lock = Lock()
    # lock=None
    readers = []
    for num in range(10):
        r, w = Pipe(duplex=False)
        readers.append(r)
        p = Process(target=f, args=(w, lock, num))
        p.start()
        w.close()

    while readers:
        for r in wait(readers):
            try:
                msg,i = r.recv()
            except EOFError:
                readers.remove(r)
            else:
                print(f'{msg} round {i}')

    print('done')