import numpy as np
import time
import cv2
import sys
import threading

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
# centro_mano = (173,251)

centro_mano = (140,160)

# Mano
# centro_mano = (91,281)

# Persona 2
# centro_mano = (165,112)

#Monitor
# centro_mano = (127,137)

# fixed_threshold = 13
# scalar = 1.67

fixed_threshold = 20
scalar = 2


def segmentate(image,bounds):
    result = np.zeros(image.shape)
    x,y,w,h, = bounds
    check_gradient = np.full(image.shape,False)
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


def segmentate_iterative(image, bounds):
    result = np.zeros(image.shape)
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
        dynamic_i_min = i_gradient/scalar
        dynamic_i_max = i_gradient*scalar
        dynamic_j_min = j_gradient/scalar
        dynamic_j_max = j_gradient*scalar
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
                i_gradient = new_gradient
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
                i_gradient = new_gradient
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
                j_gradient = new_gradient
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
                j_gradient = new_gradient
                stack.append((pixel[0],pixel[1]+1))
            else:
                #The pixel is outside the gradient
                result[pixel[0]][pixel[1]+1] = 0
    return result

def object_border(image, bounds):
    result = np.zeros(image.shape)
    y,x,h,w, = bounds
    check_gradient = np.full(image.shape,False)
    # center = (x+w/2,y+h/2)
    center = centro_mano
    # result[centro_mano[0],centro_mano[1]] = 1
    check_gradient[center[0],center[1]] = True

    gradient = 0

    pixel = center
    bot = h
    #Iterate to the bot until the gradient is too high
    while pixel[0] + 1 < h:
        pixel = (pixel[0]+1,pixel[1])
        bot = pixel
        dynamic_i_min = gradient/scalar
        dynamic_i_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
        if new_gradient < fixed_threshold or dynamic_i_min < new_gradient < dynamic_i_max:
            # result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            bot = [pixel[0]-1,pixel[1]]
            # result[pixel[0]-1][pixel[1]] = 0
            break
    # print("bot done",bot)
    pixel = center
    gradient = 0
    top = y

    #Iterate to the top until the gradient is too high
    while pixel[0] - 1 > y:

        pixel = (pixel[0]-1,pixel[1])
        top = pixel
        dynamic_i_min = gradient/scalar
        dynamic_i_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
        if new_gradient < fixed_threshold or dynamic_i_min < new_gradient < dynamic_i_max:
            # result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            top = [pixel[0]+1,pixel[1]]
            # result[pixel[0]+1][pixel[1]] = 0
            break

    # print("top done",top)

    pixel = center
    gradient = 0
    left = x

    #Iterate to the left until the gradient is too high
    while pixel[1] - 1 > x:
        pixel = (pixel[0],pixel[1]-1)
        left = pixel
        dynamic_j_min = gradient/scalar
        dynamic_j_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
        if new_gradient < fixed_threshold or dynamic_j_min < new_gradient < dynamic_j_max:
            # result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            left = [pixel[0],pixel[1]+1]
            # result[pixel[0]][pixel[1]+1] = 0
            break

    # print("left done",left)

    pixel = center
    gradient = 0
    right = w

    #Iterate to the right until the gradient is too high
    while pixel[1] + 1 < w:
        pixel = (pixel[0],pixel[1]+1)
        right = pixel
        dynamic_j_min = gradient/scalar
        dynamic_j_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
        if new_gradient < fixed_threshold or dynamic_j_min < new_gradient < dynamic_j_max:
            # result[pixel[0]][pixel[1]] = 1
            gradient = new_gradient
        else:
            right = [pixel[0],pixel[1]-1]
            # result[pixel[0]][pixel[1]-1] = 0
            break

    # print("right done",right)




    def check_borders(lock, start):
        new_gradient = 0
        stack = []
        gradient = [0,0]
        stack.append(start)
        while len(stack) > 0:
            pixel = stack.pop()
            #Check left pixel
            if pixel[1] - 1 >= x and not check_gradient[pixel[0]][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x direction was already checked
                lock.acquire()
                check_gradient[pixel[0]][pixel[1]-1] = True
                lock.release()
                if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[1] = new_gradient
                    # check the left, top and bottom of the pixel to know if it is a border pixel
                    if pixel[1] - 2 >= x:
                        left_gradient = abs(image[pixel[0]][pixel[1]-1] - image[pixel[0]][pixel[1]-2])
                    else:
                        left_gradient = fixed_threshold+1
                    if pixel[0] - 1 >= y:
                        top_gradient = abs(image[pixel[0]][pixel[1]-1] - image[pixel[0]-1][pixel[1]-1])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[0] + 1 < h:
                        bot_gradient = abs(image[pixel[0]][pixel[1]-1] - image[pixel[0]+1][pixel[1]-1])
                    else:
                        bot_gradient = fixed_threshold+1

                    if left_gradient > fixed_threshold or top_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]][pixel[1]-1] = 1
                        lock.release()
                        stack.append((pixel[0],pixel[1]-1))


            #Check top pixel
            if pixel[0] - 1 >= y and not check_gradient[pixel[0]-1][pixel[1]]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                #The gradient of the pixel in y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]-1][pixel[1]] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    # check the top, left, right of the pixel to know if it is a border pixel
                    if pixel[0] - 2 >= y:
                        top_gradient = abs(image[pixel[0]-1][pixel[1]] - image[pixel[0]-2][pixel[1]])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[1] - 1 >= x:
                        left_gradient = abs(image[pixel[0]-1][pixel[1]] - image[pixel[0]-1][pixel[1]-1])
                    else:
                        left_gradient = fixed_threshold+1
                    if pixel[1] + 1 < w:
                        right_gradient = abs(image[pixel[0]-1][pixel[1]] - image[pixel[0]-1][pixel[1]+1])
                    else:
                        right_gradient = fixed_threshold+1
                    if top_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]-1][pixel[1]] = 1
                        lock.release()
                        stack.append((pixel[0]-1,pixel[1]))

            #Check right pixel
            if pixel[1] + 1 < w and not check_gradient[pixel[0]][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x direction was already checked
                lock.acquire()
                check_gradient[pixel[0]][pixel[1]+1] = True
                lock.release()
                if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[1] = new_gradient
                    # check the right, top, bot pixels of the pixel to know if it is a border pixel
                    if pixel[1] + 2 < w:
                        right_gradient = abs(image[pixel[0]][pixel[1]+1] - image[pixel[0]][pixel[1]+2])
                    else:
                        right_gradient = fixed_threshold+1
                    if pixel[0] - 1 >= y:
                        top_gradient = abs(image[pixel[0]][pixel[1]+1] - image[pixel[0]-1][pixel[1]+1])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[0] + 1 < h:
                        bot_gradient = abs(image[pixel[0]][pixel[1]+1] - image[pixel[0]+1][pixel[1]+1])
                    else:
                        bot_gradient = fixed_threshold+1

                    if right_gradient > fixed_threshold or top_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]][pixel[1]+1] = 1
                        lock.release()
                        stack.append((pixel[0],pixel[1]+1))

            #Check bottom pixel
            if pixel[0] + 1 < h and not check_gradient[pixel[0]+1][pixel[1]]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                #The gradient of the pixel in y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]+1][pixel[1]] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    # check the bottom, left, right pixels of the pixel to know if it is a border pixel
                    if pixel[0] + 2 < h:
                        bot_gradient = abs(image[pixel[0]+1][pixel[1]] - image[pixel[0]+2][pixel[1]])
                    else:
                        bot_gradient = fixed_threshold+1
                    if pixel[1] - 1 >= x:
                        left_gradient = abs(image[pixel[0]+1][pixel[1]] - image[pixel[0]+1][pixel[1]-1])
                    else:
                        left_gradient = fixed_threshold+1
                    if pixel[1] + 1 < w:
                        right_gradient = abs(image[pixel[0]+1][pixel[1]] - image[pixel[0]+1][pixel[1]+1])
                    else:
                        right_gradient = fixed_threshold+1

                    if bot_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]+1][pixel[1]] = 1
                        lock.release()
                        stack.append((pixel[0]+1,pixel[1]))

            # check top left corner
            if pixel[0] - 1 >= y and pixel[1] - 1 >= x and not check_gradient[pixel[0]-1][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]-1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]-1][pixel[1]-1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the top, left, right pixels of the pixel to know if it is a border pixel

                    if pixel[0] - 2 >= y:
                        top_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-2][pixel[1]-1])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[1] - 2 >= x:
                        left_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-1][pixel[1]-2])
                    else:
                        left_gradient = fixed_threshold+1

                    right_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-1][pixel[1]])

                    bot_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])

                    if top_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]-1][pixel[1]-1] = 1
                        lock.release()
                        stack.append((pixel[0]-1,pixel[1]-1))

            # check top right corner
            if pixel[0] - 1 >= y and pixel[1] + 1 < w and not check_gradient[pixel[0]-1][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]+1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]-1][pixel[1]+1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the top, left, right of the pixel to know if it is a border pixel
                    if pixel[0] - 2 >= y:
                        top_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-2][pixel[1]+1])
                    else:
                        top_gradient = fixed_threshold+1

                    left_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-1][pixel[1]])

                    if pixel[1] + 2 < w:
                        right_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-1][pixel[1]+2])
                    else:
                        right_gradient = fixed_threshold+1

                    bot_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]][pixel[1]+1])

                    if top_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]-1][pixel[1]+1] = 1
                        lock.release()
                        stack.append((pixel[0]-1,pixel[1]+1))

            # check bottom left corner
            if pixel[0] + 1 < h and pixel[1] - 1 >= x and not check_gradient[pixel[0]+1][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]-1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]+1][pixel[1]-1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the bot, left, right of the pixel to know if it is a border pixel
                    bot_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])

                    if pixel[1] - 2 >= x:
                        left_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]+1][pixel[1]-2])
                    else:
                        left_gradient = fixed_threshold+1

                    right_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]+1][pixel[1]])

                    top_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])

                    if bot_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or top_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]+1][pixel[1]-1] = 1
                        lock.release()
                        stack.append((pixel[0]+1,pixel[1]-1))
            # check bottom right corner
            if pixel[0] + 1 < h and pixel[1] + 1 < w and not check_gradient[pixel[0]+1][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]+1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]+1][pixel[1]+1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the bot, left, right of the pixel to know if it is a border pixel
                    if pixel[0] + 2 < h:
                        bot_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+2][pixel[1]+1])
                    else:
                        bot_gradient = fixed_threshold+1

                    left_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+1][pixel[1]])

                    if pixel[1] + 2 < w:
                        right_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+1][pixel[1]+2])
                    else:
                        right_gradient = fixed_threshold+1

                        top_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]][pixel[1]+1])

                    if bot_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or top_gradient > fixed_threshold:
                        lock.acquire()
                        result[pixel[0]+1][pixel[1]+1] = 1
                        lock.release()
                        stack.append((pixel[0]+1,pixel[1]+1))

    lock = threading.Lock()

    #create threads
    t1 = threading.Thread(target=check_borders, args=(lock, top))
    t2 = threading.Thread(target=check_borders, args=(lock, bot))
    t3 = threading.Thread(target=check_borders, args=(lock, left))
    t4 = threading.Thread(target=check_borders, args=(lock, right))

    #start threads
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    #wait for threads to finish
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    return result


if __name__ == "__main__":
    # Import Depth Matrix
    data= np.loadtxt("src/depth/person1.csv", delimiter=",", skiprows=0, dtype=np.float32)
    t1 = time.time()
    img1 = segmentate(data, (0,0,240,320))
    t2 = time.time()
    img = object_border(data, (0,0,240,320))
    t3 = time.time()
    img1 = segmentate_iterative(data, (0,0,240,320))
    t4 = time.time()
    print("Segmentation recursive time: ", t2-t1)
    print("Segmentation iterative time: ", t4-t3)
    print("Segmentation border time: ", t3-t2)

    # border time is % faster than recursive
    print(f"Border is {((t2-t1)/(t3-t2))*100:.2f}% faster than recursive ({((t2-t1)/(t3-t2)):.2f} times faster)")
    print(f"Border is {((t4-t3)/(t3-t2))*100:.2f}% faster than iterative ({((t4-t3)/(t3-t2)):.2f} times faster)")

    # show_img(data)
    save_img(img1, "src/img/person1.png")
    save_img(img, "src/img/person1_border.png")
    save_img(data, "src/img/person1_depth.png")
    np.savetxt("src/img/person1_segmented.csv", img, delimiter=",", fmt='%d')