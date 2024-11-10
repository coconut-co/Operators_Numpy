import cv2
import numpy as np

def sobel_filter(axis):
    if axis == 'x':
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    elif axis == 'y':
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
def convolution2D(image, kernel, padding=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    #对图像进行填充    填充模式为constant，填充值为0
    image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    #np.zeros()， np.zeros 需要一个形状参数，这是一个包含两个元素的元组（）
    #输出图像尺寸，(intput_height-kernel_height + 2*padding + 1)/stride + 1
    output = np.zeros((image_height-kernel_height + 2*padding + 1, 
                      image_width-kernel_width + 2*padding + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)


    return output

if __name__ == "__main__":
    sobel_x = sobel_filter('x')
    sobel_y = sobel_filter('y')

    kernel = np.ones((3, 3), dtype=np.float32) / (3 * 3)

    image = cv2.imread("test.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    output_x = convolution2D(image, sobel_x)
    output_y = convolution2D(image, sobel_y)

    magnitude = np.sqrt(output_x**2 + output_y**2)               # 梯度幅值
    output = np.clip(magnitude, 0, 255).astype(np.uint8)         # 归一化到[0, 255]，并转换为uint8  
    print(output.shape)


    cv2.imwrite("dst.jpg", output)