import numpy as np 
import time
import cv2
import sys

sys.setrecursionlimit(76800)

def show_img (img):
    # Changes format from float to uint8 as gradient
    uint_img = np.array(img*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Image",grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def save_img (img, name):
    # Changes format from float to uint8 as gradient
    uint_img = np.array(img*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(name,grayImage)
    return None

#########################################
# Import Depth Matrix 
# datas= np.loadtxt("matrix.csv", delimiter=",", skiprows=0)

# PersonaSentada
centro_mano = (173,251)

# Mano
# centro_mano = (91,281)

# Persona 2
# centro_mano = (165,112)

#Monitor
# centro_mano = (127,137)

fixed_threshold = 100
scalar = 2

def segmentate(image,bounds):
    result = np.zeros(image.shape)
    height, width = image.shape
    x,y,w,h, = bounds
    check_gradient = np.full(image.shape,False)
    # center = (x+w/2,y+h/2)
    center = centro_mano
    result[centro_mano[0],centro_mano[1]] = 1
    check_gradient[center[0],center[1]] = True

    def check_pixel(pixel,gradient):
        i_gradient = gradient[0]
        j_gradient = gradient[1]
        dynamic_i_min =i_gradient/scalar
        dynamic_i_max =i_gradient*scalar
        dynamic_j_min =j_gradient/scalar
        dynamic_j_max =j_gradient*scalar
        #Check right pixel
        if pixel[0] + 1 < x+w and not check_gradient[pixel[0]+1][pixel[1]]:
            #The pixel is inside the bounds and has not been checked
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
            #The gradient of the pixel in x direction was already checked
            check_gradient[pixel[0]+1][pixel[1]] = True
            if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                result[pixel[0]+1][pixel[1]] = 1
                check_pixel((pixel[0]+1,pixel[1]),(new_gradient,j_gradient))
            else:
                #The pixel is outside the gradient
                result[pixel[0]+1][pixel[1]] = 0

        #Check left pixel
        if pixel[0] - 1 > x and not check_gradient[pixel[0]-1][pixel[1]]:
            #The pixel is inside the bounds and has not been checked
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
            #The gradient of the pixel in x direction was already checked
            check_gradient[pixel[0]-1][pixel[1]] = True
            if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                result[pixel[0]-1][pixel[1]] = 1
                check_pixel((pixel[0]-1,pixel[1]),(new_gradient,j_gradient))
            else:
                #The pixel is outside the gradient
                result[pixel[0]-1][pixel[1]] = 0

        #Check top pixel
        if pixel[1] - 1 > y and not check_gradient[pixel[0]][pixel[1]-1]:
            #The pixel is inside the bounds and has not been checked
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
            #The gradient of the pixel in x direction was already checked
            check_gradient[pixel[0]][pixel[1]-1] = True
            if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                result[pixel[0]][pixel[1]-1] = 1
                check_pixel((pixel[0],pixel[1]-1),(i_gradient,new_gradient))
            else:
                #The pixel is outside the gradient
                result[pixel[0]][pixel[1]-1] = 0

        #Check bottom pixel
        if pixel[1] + 1 < y+h and not check_gradient[pixel[0]][pixel[1]+1]:
            #The pixel is inside the bounds and has not been checked
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
            #The gradient of the pixel in x direction was already checked
            check_gradient[pixel[0]][pixel[1]+1] = True
            if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                result[pixel[0]][pixel[1]+1] = 1
                check_pixel((pixel[0],pixel[1]+1),(i_gradient,new_gradient))
            else:
                #The pixel is outside the gradient
                result[pixel[0]][pixel[1]+1] = 0

    check_pixel(center,(0,0))
    return result

if __name__ == "__main__":
    t1 = time.time()
    img = segmentate(data, (0,0,240,320))
    t2 = time.time()
    print(t2-t1)
    # show_img(data)
    show_img(img)