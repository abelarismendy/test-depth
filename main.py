import os, platform, subprocess, re
from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer
import depth
import csv

DEPTH_FOLDER = "src/depth"
OUTPUT_FOLDER = "src/output"
IMG_FOLDER = "src/img"
TESTS_FOLDER = "tests"
TEST_FILENAME = "results.csv"
N = 10

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

# # normal function to run on cpu
# def func(a):
#     for i in range(10000000):
#         a[i]+= 1

# # function optimized to run on gpu
# @jit(target_backend='cuda')
# def func2(a):
#     for i in range(10000000):
#         a[i]+= 1

# def simple_gpu_test():
#     n = 10000000
#     a = np.ones(n, dtype = np.float64)
#     start = timer()
#     func(a)
#     print("without GPU:", timer()-start)
#     start = timer()
#     func2(a)
#     print("with GPU:", timer()-start)


def test_cpu(data):
    img = depth.segmentate(data, (0,0,240,320))
    return img



def main():
    proccesor = get_processor_name().strip()
    gpu = get_gpu_name()
    os_name = platform.system()
    os_version = platform.platform()
    method = "our_algorithm"
    print("Processor: ", proccesor)
    print("GPU: ", gpu)
    print("OS: ", os_name)
    print("OS version: ", os_version)

    header = ["processor", "gpu", "os", "os_version", "file", "time_avg", "n", "method"]
    # save results to csv file in the folder tests
    if not os.path.exists(TESTS_FOLDER):
        os.makedirs(TESTS_FOLDER)
    if not os.path.exists(TESTS_FOLDER + "/" + TEST_FILENAME):
        f = open(TESTS_FOLDER+"/"+TEST_FILENAME, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
    else:
        f = open(TESTS_FOLDER+"/"+TEST_FILENAME, "a", newline="")
        writer = csv.writer(f)

    # get all csv files in the folder DEPTH_FOLDER
    files = os.listdir(DEPTH_FOLDER)
    csv_files = [file for file in files if file.endswith(".csv")]
    for file in csv_files:
        total = 0
        for i in range(N):
            data = np.loadtxt(DEPTH_FOLDER+"/"+file, delimiter=",")
            start = timer()
            img = test_cpu(data)
            end = timer()
            total += end - start
        avg = total / N
        print("Time for file: ", file, " is: ", avg)
        test = [proccesor, gpu, os_name, os_version, file, avg, N, method]
        writer.writerow(test)
        # save image to the folder OUTPUT_FOLDER
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        np.savetxt(OUTPUT_FOLDER+"/"+method+file, img, delimiter=",", fmt="%d")

    f.close()
    transform_csv_to_img()

def transform_csv_to_img():
    files = os.listdir(OUTPUT_FOLDER)
    csv_files = [file for file in files if file.endswith(".csv")]
    for file in csv_files:
        data = np.loadtxt(OUTPUT_FOLDER+"/"+file, delimiter=",")
        depth.save_img(data, IMG_FOLDER+"/"+file[:-4]+".png")

if __name__ == '__main__':
    main()
