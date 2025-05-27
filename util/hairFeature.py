import cv2
import numpy as np

def count_white_percentage(image, threshold=240):
    """
    Counts the number of white pixels in a grayscale image.

    :param image: the image
    :param threshold: (int) Intensity threshold to consider a pixel as "white".

    
    :returns: Percentage of white pixels in the image.
    """
    # Load the image in grayscale
    # Create a mask of pixels above the threshold
    white_mask = image >= threshold

    # Count white pixels
    white_pixel_count = np.sum(white_mask)
    total_pixels = image.size
    white_percentage = (white_pixel_count / total_pixels)

    return white_percentage

def getHair(img_org, kernel_size=25, threshold=10):
    # kernel for the morphological filtering
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # perform the hat transform on the grayscale image
    hat_img = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel) 
    
    
    # threshold the hair contours
    _, thresh = cv2.threshold(hat_img, threshold, 255, cv2.THRESH_BINARY)
    
    return thresh


def amountOfHairFeature(img_org, black_threshold: int = 50) -> float:
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(img_gray, black_threshold, 255, cv2.THRESH_BINARY)

    count_black_pxls = np.sum(thresh == 0)

    return (count_black_pxls / thresh.size) * 10


if __name__ == "__main__":
    # test with random noise
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    print(amountOfHairFeature(image))



