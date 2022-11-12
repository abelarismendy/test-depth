import os, platform, subprocess, re
from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

def get_gpu_name():
    try:
        name = cuda.gpus[0]
    except:
        # name = input("Enter GPU name: ")
        name = "N/A"
    return name

# normal function to run on cpu
def func(a):
    for i in range(10000000):
        a[i]+= 1

# function optimized to run on gpu
@jit(target_backend='cuda')
def func2(a):
    for i in range(10000000):
        a[i]+= 1

def simple_gpu_test():
    n = 10000000
    a = np.ones(n, dtype = np.float64)
    start = timer()
    func(a)
    print("without GPU:", timer()-start)
    start = timer()
    func2(a)
    print("with GPU:", timer()-start)

def main():
    proccesor = get_processor_name()
    gpu = get_gpu_name()
    os = platform.system()
    os_version = platform.platform()
    print("Processor: ", proccesor)
    print("GPU: ", gpu)
    print("OS: ", os)
    print("OS version: ", os_version)

    simple_gpu_test()


if __name__ == '__main__':
    main()
