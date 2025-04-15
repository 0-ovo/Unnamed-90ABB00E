#!/usr/bin/python3
# -*- coding: utf-8 -*-                                          
# This file is part of the Image Processing Library.
# 由于我在多台电脑上运行paddleocr复原文档的效果不如预期,所以使用opencv手动变换图片.
# 该文件主要用于图片处理,使用python实现.
# 大部分注释为自动生成,我只作了函数注解...
# 实际上函数注解是为了方便代码自动补全及着色,但也支持启用类型检查
# 由於輸入法抽風,部分注釋為繁體中文
import cv2
import numpy as np
from numpy.typing import NDArray



def crop_image(image_path : str | cv2.typing.MatLike, points : NDArray[np.float32]) -> cv2.typing.MatLike:
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
    if isinstance(image_path, np.ndarray):
        # If image_path is already an image, use it directly
        image = image_path
    else:
        # Otherwise, read the image from the file path
        image = cv2.imread(image_path)
        

    if not isinstance(points, np.ndarray): # type: ignore
        points = np.float32(points)
            
    if len(points) != 4:
        raise ValueError("points should be a list of 4 points.")
    # Crop the image
    imageshape : tuple[int,int] = image.shape

    M : cv2.typing.MatLike = cv2.getPerspectiveTransform(points,np.array([[0,0],[imageshape[1],0],[0,imageshape[0]],[imageshape[1],imageshape[0]]]))
    cropped_image : cv2.typing.MatLike = cv2.warpPerspective(image, M, (imageshape[1], imageshape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return cropped_image

def resize(src : cv2.typing.MatLike,fx : float,fy : float,interpolation : int=cv2.INTER_LINEAR) -> cv2.typing.MatLike:
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
def filterimg(src : cv2.typing.MatLike | str, colorRange : NDArray[np.float32] = np.array(((176,100,38),(180,255,255)), dtype=np.float32)) -> cv2.typing.MatLike:
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
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    print(colorRange[0])
    # Create a mask based on the color range
    mask = cv2.inRange(hsv, colorRange[0], colorRange[1])
    
    # Apply the mask to the original image
    masked = cv2.bitwise_and(src, src, mask=mask)
    # Convert to grayscale
    
    result = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    return result

def showimg(src : cv2.typing.MatLike | str) -> None:
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
    
    cv2.imshow("Image", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

