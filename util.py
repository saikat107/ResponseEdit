import inspect
import os

def debug(*msg):
    import inspect
    file_path = inspect.stack()[1][1]
    line_num = inspect.stack()[1][2]
    file_name = file_path
    if os.getcwd() in file_path:
        file_name = file_path[len(os.getcwd())+1:]
    stack = str(file_name) + ' # ' + str(line_num)
    print(stack, end=' ')
    res = '\t'
    for ms in msg:
        res += (str(ms) + ' ')
    print(res)

if __name__ == '__main__':
    debug('hello')

