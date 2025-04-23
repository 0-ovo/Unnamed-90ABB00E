# -*- coding: utf-8 -*-                                          
# This file is part of the Image Processing Library.
# 由于我在多台电脑上运行paddleocr复原文档的效果不如预期,所以使用opencv手动变换图片.
# 该文件主要用于图片处理,使用python实现.
# 非OOP編程,請導入需要的函數
# 大部分注释为自动生成,我只作了函数注解...
# 实际上函数注解是为了方便代码自动补全及着色,但也支持启用类型检查
# 由於輸入法抽風,部分注釋為繁體中文
import cv2
import numpy as np
from numpy.typing import NDArray
import PIL.Image # 為防止命名衝突,未采取 from PIL import Image

def convertToPIL(src : cv2.typing.MatLike) -> PIL.Image.Image:
    """
    Convert a cv2 image to a PIL image.
    
    Parameters:
        src (cv2.typing.MatLike): The source image to be converted.
        
    Returns:
        PIL.Image.Image: The converted PIL image.
    """
    return PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))


def convertToCV2(src : PIL.Image.Image) -> cv2.typing.MatLike:
    """
    Convert a PIL image to a cv2 image.
    
    Parameters:
        src (PIL.Image.Image): The source image to be converted.
        
    Returns:
        cv2.typing.MatLike: The converted cv2 image.
    """
    return cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)


def cropImage(src : str | cv2.typing.MatLike, points : NDArray[np.float32]) -> cv2.typing.MatLike:
    """
    Crop an image to the specified rectangle.
    
    Parameters:
        image_path(str | cv2.typing.MatLike): Path to the input image.(or an image object)
        points(np.ndarray): A tuple or list containing the coordinates of the rectangle to crop.
            The format should be (top_left(x,y), top_right(x,y),lower_left(x,y),lower_right(x,y)). 
    
    Returns:
    - cropped_image: The cropped image.
    """

    # Read the image
    # 手工注释：此段代码由copilot生成，有些冗长，在后续这会化简为 src = cv2.imread(src) if isinstance(src, str) else src
    if isinstance(src, np.ndarray):
        # If image_path is already an image, use it directly
        src = src
    else:
        # Otherwise, read the image from the file path
        src = cv2.imread(src)
        
        
    #手工注释:这是为了冗余(尝试将错误的传入转为numpy数组)
    if not isinstance(points, np.ndarray): # type: ignore
        points = np.array(points, dtype=np.float32)
            
    if len(points) != 4:
        raise ValueError("points should be a list of 4 points.")
    # Crop the image
    imageshape : tuple[int,int] = src.shape

    M : cv2.typing.MatLike = cv2.getPerspectiveTransform(points,np.array([[0,0],[imageshape[1],0],[0,imageshape[0]],[imageshape[1],imageshape[0]]], dtype=np.float32))
    cropped_image : cv2.typing.MatLike = cv2.warpPerspective(src, M, (imageshape[1], imageshape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return cropped_image


def resizeImage(src : cv2.typing.MatLike,fx : float,fy : float,interpolation : int=cv2.INTER_LINEAR) -> cv2.typing.MatLike:
    """
    Resize an image using OpenCV.
    
    Parameters:
        src (cv2.typing.MatLike): The source image to be resized.
        fx (float): Scale factor along the horizontal axis.
        fy (float): Scale factor along the vertical axis.
        interpolation (int, optional): Interpolation method. Default is cv2.INTER_LINEAR.
        
    Returns:
        cv2.typing.MatLike: The resized image.
    """
    return cv2.resize(src, None, fx=fx, fy=fy, interpolation=interpolation)


# 手動注釋: 該函數用於從圖片中提取特定hsv顔色
def filterImageColor(src : cv2.typing.MatLike | str, colorRange : NDArray[np.float32] = np.array(((176,100,38),(180,255,255)), dtype=np.float32)) -> cv2.typing.MatLike:
    """
    Filter an image based on a color range.
    
    Parameters:
        src (cv2.typing.Matlike): The source image to be filtered.
        colorRange (np.ndarray): The color range for filtering.
        
    Returns:
        cv2.typing.MatLike: The filtered image.
        
    Raises:
        ValueError: If the image is not valid.
        
    """
    
    src = cv2.imread(src) if isinstance(src, str) else src
    
    if src is None: # type: ignore
        raise ValueError("Invalid image.")
    
    
    # Convert the image to HSV color space
    hsv : cv2.typing.MatLike = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on the color range
    mask : cv2.typing.MatLike = cv2.inRange(hsv, colorRange[0], colorRange[1])
    
    # Apply the mask to the original image
    masked : cv2.typing.MatLike = cv2.bitwise_and(src, src, mask=mask)
    
    

    
    return cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

def cutImage(src : cv2.typing.MatLike | str, ):
    raise IndentationError("cutimg is not implemented yet.")
    pass

def showImage(src : cv2.typing.MatLike | str , windowname : str = "IMG") -> None:
    """
    Display an image using OpenCV.
    
    Parameters:
        src (cv2.typing.MatLike | str): The source image to be displayed.
        
    Returns:
        None
    """
    src = cv2.imread(src) if isinstance(src, str) else src
    
    if src is None: # type: ignore
        raise ValueError("Invalid image.")
    
    cv2.imshow(windowname, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def medianFilterImage(src : cv2.typing.MatLike | str,noise : int = 3000, coresize : int = 5) -> cv2.typing.MatLike:
    """
    Add noise to an image and apply median filtering.
    Parameters:
        src (cv2.typing.MatLike | str): The source image to be filtered.
        noise (int): The number of noise points to add. Default is 3000.
        coresize (int): The size of the median filter kernel. Default is 5.
    Returns:
        cv2.typing.MatLike: The filtered image.
    """

    # 手动注释:你可能注意到此时赋值调用了copy()方法，这是为了防止对原始图片进行更改
    src = cv2.imread(src) if isinstance(src, str) else src.copy()

    
    if src is None: # type: ignore
        raise ValueError("Invalid image.")

    h : int = src.shape[0]
    w : int = src.shape[1]

    # 添加噪声
    for noise_index in range(noise): # type: ignore
        x : int= np.random.randint(0, h) 
        y : int= np.random.randint(0, w) 
        src[x, y] = 255

    # 中值滤波
    return cv2.medianBlur(src, coresize)


