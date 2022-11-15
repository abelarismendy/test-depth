import numpy as np 
import time
import cv2
import sys
from numba import jit, cuda

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


# PersonaSentada
centro_mano = (173,251)

# Mano
# centro_mano = (91,281)

# Persona 2
# centro_mano = (165,112)

#Monitor
# centro_mano = (127,137)

fixed_threshold = 13
scalar = 1.65


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

# @jit(target_backend = 'cuda', nopython = True)
def segmentate_iterative(image, bounds):
    result = np.zeros(image.shape)
    height, width = image.shape
    x,y,w,h, = bounds
    check_gradient = np.full(image.shape,False)
    # center = (x+w/2,y+h/2)
    center = centro_mano
    result[centro_mano[0],centro_mano[1]] = 1
    check_gradient[center[0],center[1]] = True
    stack = [center]
    i_gradient = 0
    j_gradient = 0
    dynamic_i_min = 0
    dynamic_i_max = 0
    dynamic_j_min = 0
    dynamic_j_max = 0

    while len(stack) > 0:
        pixel = stack.pop()
        #Check right pixel
        if pixel[0] + 1 < x+w and not check_gradient[pixel[0]+1][pixel[1]]:
            #The pixel is inside the bounds and has not been checked
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
            #The gradient of the pixel in x direction was already checked
            check_gradient[pixel[0]+1][pixel[1]] = True
            if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                result[pixel[0]+1][pixel[1]] = 1
                stack.append((pixel[0]+1,pixel[1]))
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
                stack.append((pixel[0]-1,pixel[1]))
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
                stack.append((pixel[0],pixel[1]-1))
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
                stack.append((pixel[0],pixel[1]+1))
            else:
                #The pixel is outside the gradient
                result[pixel[0]][pixel[1]+1] = 0
    return result

def object_border(image, bounds):
    result = np.zeros(image.shape)
    height, width = image.shape
    x,y,w,h, = bounds
    check_gradient = np.full(image.shape,False)
    # center = (x+w/2,y+h/2)
    center = centro_mano
    result[centro_mano[0],centro_mano[1]] = 1
    check_gradient[center[0],center[1]] = True

    gradient = 0

    pixel = center
    bot = center
    #Iterate to the top until the gradient is too high
    while pixel[0] + 1 < x+w:

        pixel = (pixel[0]+1,pixel[1])
        dynamic_i_min = gradient/scalar
        dynamic_i_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
        if new_gradient < fixed_threshold or dynamic_i_min < new_gradient < dynamic_i_max:
            result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            bot = pixel
            break

    pixel = center
    gradient = 0
    top = center

    #Iterate to the bottom until the gradient is too high
    while pixel[0] - 1 > x:

        pixel = (pixel[0]-1,pixel[1])
        dynamic_i_min = gradient/scalar
        dynamic_i_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
        if new_gradient < fixed_threshold or dynamic_i_min < new_gradient < dynamic_i_max:
            result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            top = pixel
            break

    pixel = center
    gradient = 0
    left = center

    #Iterate to the left until the gradient is too high
    while pixel[1] - 1 > y:
        pixel = (pixel[0],pixel[1]-1)
        dynamic_j_min = gradient/scalar
        dynamic_j_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
        if new_gradient < fixed_threshold or dynamic_j_min < new_gradient < dynamic_j_max:
            result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            left = pixel
            break

    pixel = center
    gradient = 0
    right = center

    #Iterate to the right until the gradient is too high
    while pixel[1] + 1 < y+h:
        pixel = (pixel[0],pixel[1]+1)
        dynamic_j_min = gradient/scalar
        dynamic_j_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
        if new_gradient < fixed_threshold or dynamic_j_min < new_gradient < dynamic_j_max:
            result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            right = pixel
            break

    stack = []
    gradient = [0,0]
    stack.append(top)
    while len(stack) > 0:
        pixel = stack.pop()
        if pixel[0] + 1 < x+w and not check_gradient[pixel[0]+1][pixel[1]]:
            #The pixel is inside the bounds and has not been checked
            dynamic_i_min = gradient[0]/scalar
            dynamic_i_max = gradient[0]*scalar
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
            gradient[0] = new_gradient
            #The gradient of the pixel in y direction was already checked
            check_gradient[pixel[0]+1][pixel[1]] = True
            if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                #check if the pixel is on the border
                try:
                    top_gradient = abs(image[pixel[0]+2][pixel[1]] - image[pixel[0]+1][pixel[1]])
                    dynamic_top_min = top_gradient/scalar
                    dynamic_top_max = top_gradient*scalar
                except:
                    top_gradient = 0
                    dynamic_top_min = 0
                    dynamic_top_max = 0
                try:
                    right_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+1][pixel[1]])
                    dynamic_right_min = right_gradient/scalar
                    dynamic_right_max = right_gradient*scalar
                except:
                    right_gradient = 0
                    dynamic_right_min = 0
                    dynamic_right_max = 0

                try:
                    left_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]+1][pixel[1]])
                    dynamic_left_min = left_gradient/scalar
                    dynamic_left_max = left_gradient*scalar
                except:
                    left_gradient = 0
                    dynamic_left_min = 0
                    dynamic_left_max = 0

                if top_gradient > fixed_threshold or right_gradient > fixed_threshold or left_gradient > fixed_threshold or dynamic_top_min < top_gradient < dynamic_top_max or dynamic_right_min < right_gradient < dynamic_right_max or dynamic_left_min < left_gradient < dynamic_left_max:
                    result[pixel[0]+1][pixel[1]] = 1
                    stack.append((pixel[0]+1,pixel[1]))
            # else:
            #     #The pixel is outside the gradient
            #     result[pixel[0]+1][pixel[1]] = 0
        if pixel[0] - 1 > x and not check_gradient[pixel[0]-1][pixel[1]]:
            #The pixel is inside the bounds and has not been checked
            dynamic_i_min = gradient[0]/scalar
            dynamic_i_max = gradient[0]*scalar
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
            gradient[0] = new_gradient
            #The gradient of the pixel in y direction was already checked
            check_gradient[pixel[0]-1][pixel[1]] = True
            if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                #check if the pixel is on the border
                try:
                    bot_gradient = abs(image[pixel[0]-2][pixel[1]] - image[pixel[0]-1][pixel[1]])
                    dynamic_bot_min = bot_gradient/scalar
                    dynamic_bot_max = bot_gradient*scalar
                except:
                    bot_gradient = 0
                    dynamic_bot_min = 0
                    dynamic_bot_max = 0
                try:
                    right_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-1][pixel[1]])
                    dynamic_right_min = right_gradient/scalar
                    dynamic_right_max = right_gradient*scalar
                except:
                    right_gradient = 0
                    dynamic_right_min = 0
                    dynamic_right_max = 0
                
                try:
                    left_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-1][pixel[1]])
                    dynamic_left_min = left_gradient/scalar
                    dynamic_left_max = left_gradient*scalar
                except:
                    left_gradient = 0
                    dynamic_left_min = 0
                    dynamic_left_max = 0

                if bot_gradient > fixed_threshold or right_gradient > fixed_threshold or left_gradient > fixed_threshold or dynamic_bot_min < bot_gradient < dynamic_bot_max or dynamic_right_min < right_gradient < dynamic_right_max or dynamic_left_min < left_gradient < dynamic_left_max:
                    result[pixel[0]-1][pixel[1]] = 1
                    stack.append((pixel[0]-1,pixel[1]))
            # else:
            #     #The pixel is outside the gradient
            #     result[pixel[0]-1][pixel[1]] = 0
        if pixel[1] + 1 < y+h and not check_gradient[pixel[0]][pixel[1]+1]:
            #The pixel is inside the bounds and has not been checked
            dynamic_j_min = gradient[1]/scalar
            dynamic_j_max = gradient[1]*scalar
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
            gradient[1] = new_gradient
            #The gradient of the pixel in y direction was already checked
            check_gradient[pixel[0]][pixel[1]+1] = True
            if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                #check if the pixel is on the border
                try:
                    top_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]][pixel[1]+1])
                    dynamic_top_min = top_gradient/scalar
                    dynamic_top_max = top_gradient*scalar
                except:
                    top_gradient = 0
                    dynamic_top_min = 0
                    dynamic_top_max = 0
                try:
                    right_gradient = abs(image[pixel[0]][pixel[1]+2] - image[pixel[0]][pixel[1]+1])
                    dynamic_right_min = right_gradient/scalar
                    dynamic_right_max = right_gradient*scalar
                except:
                    right_gradient = 0
                    dynamic_right_min = 0
                    dynamic_right_max = 0
                try:
                    bot_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]][pixel[1]+1])
                    dynamic_bot_min = bot_gradient/scalar
                    dynamic_bot_max = bot_gradient*scalar
                except:
                    bot_gradient = 0
                    dynamic_bot_min = 0
                    dynamic_bot_max = 0

                if top_gradient > fixed_threshold or right_gradient > fixed_threshold or bot_gradient > fixed_threshold or dynamic_top_min < top_gradient < dynamic_top_max or dynamic_right_min < right_gradient < dynamic_right_max or dynamic_bot_min < bot_gradient < dynamic_bot_max:
                    result[pixel[0]][pixel[1]+1] = 1
                    stack.append((pixel[0],pixel[1]+1))
            # else:
            #     #The pixel is outside the gradient
            #     result[pixel[0]][pixel[1]+1] = 0
        if pixel[1] - 1 > y and not check_gradient[pixel[0]][pixel[1]-1]:
            #The pixel is inside the bounds and has not been checked
            dynamic_j_min = gradient[1]/scalar
            dynamic_j_max = gradient[1]*scalar
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
            gradient[1] = new_gradient
            #The gradient of the pixel in y direction was already checked
            check_gradient[pixel[0]][pixel[1]-1] = True
            if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                #check if the pixel is on the border
                try:
                    top_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])
                    dynamic_top_min = top_gradient/scalar
                    dynamic_top_max = top_gradient*scalar
                except:
                    top_gradient = 0
                    dynamic_top_min = 0
                    dynamic_top_max = 0
                try:
                    left_gradient = abs(image[pixel[0]][pixel[1]-2] - image[pixel[0]][pixel[1]-1])
                    dynamic_left_min = left_gradient/scalar
                    dynamic_left_max = left_gradient*scalar
                except:
                    left_gradient = 0
                    dynamic_left_min = 0
                    dynamic_left_max = 0
                try:
                    bot_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])
                    dynamic_bot_min = bot_gradient/scalar
                    dynamic_bot_max = bot_gradient*scalar
                except:
                    bot_gradient = 0
                    dynamic_bot_min = 0
                    dynamic_bot_max = 0


                if top_gradient > fixed_threshold or left_gradient > fixed_threshold or bot_gradient > fixed_threshold or dynamic_top_min < top_gradient < dynamic_top_max or dynamic_left_min < left_gradient < dynamic_left_max or dynamic_bot_min < bot_gradient < dynamic_bot_max:
                    result[pixel[0]][pixel[1]-1] = 1
                    stack.append((pixel[0],pixel[1]-1))
            # else:
            #     #The pixel is outside the gradient
            #     result[pixel[0]][pixel[1]-1] = 0

    return result




@jit(target_backend = 'cuda', nopython = True)
def segmentate_gpu(image, bounds):
    result = np.zeros(image.shape)
    height, width = image.shape
    x,y,w,h, = bounds
    check_gradient = np.full(image.shape,False)
    # center = (x+w/2,y+h/2)
    center = centro_mano
    result[centro_mano[0],centro_mano[1]] = 1
    check_gradient[center[0],center[1]] = True
    stack = [center]
    i_gradient = 0
    j_gradient = 0
    dynamic_i_min = 0
    dynamic_i_max = 0
    dynamic_j_min = 0
    dynamic_j_max = 0

    while len(stack) > 0:
        pixel = stack.pop()
        #Check right pixel
        if pixel[0] + 1 < x+w and not check_gradient[pixel[0]+1][pixel[1]]:
            #The pixel is inside the bounds and has not been checked
            new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
            #The gradient of the pixel in x direction was already checked
            check_gradient[pixel[0]+1][pixel[1]] = True
            if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                #The pixel is inside the gradient and respects or is below the fixed threshold
                result[pixel[0]+1][pixel[1]] = 1
                stack.append((pixel[0]+1,pixel[1]))
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
                stack.append((pixel[0]-1,pixel[1]))
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
                stack.append((pixel[0],pixel[1]-1))
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
                stack.append((pixel[0],pixel[1]+1))
            else:
                #The pixel is outside the gradient
                result[pixel[0]][pixel[1]+1] = 0
    return result


if __name__ == "__main__":
    # Import Depth Matrix
    data= np.loadtxt("src/depth/0.csv", delimiter=",", skiprows=0)
    t1 = time.time()
    # img = segmentate_iterative(data, (0,0,240,320))
    img = object_border(data, (0,0,240,320))
    t2 = time.time()
    print(t2-t1)
    # show_img(data)
    save_img(img, "src/img/0.png")
    save_img(data, "src/img/0_depth.png")