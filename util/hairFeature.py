import cv2
import numpy as np
def count_white_percentage(image, threshold=240):
    """
    Counts the number of white pixels in a grayscale image.

    Parameters:
        image: the image
        threshold (int): Intensity threshold to consider a pixel as "white".

    Returns:
        float: Percentage of white pixels in the image.
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
def amountOfHairFeature(img_org):
    #extract hair
    hair = getHair(img_org)
    # get the amount of white pixels
    return count_white_percentage(hair)
if __name__ == "__main__":
    # test with random noise
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    print(amountOfHairFeature(image))



