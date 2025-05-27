import img_util

def extract_mask(img_path, mask_path: str | None = None):
    """Provided an image path, it will get both the image and mask and return the processed image."""
    if mask_path is None:
        mask_path = get_mask_path(img_path)
    
    assert isinstance(mask_path, str)

    img = img_util.readImageFile(img_path)[0]
    mask = img_util.readImageFile(mask_path)[0]
    img[mask==0] = 0
    return img

def get_mask_path(img_path):
    """This implementation assumes the directory structure looks like './data/' with two subdirectories './data/images' and './data/masks'."""
    return "./data/masks/"+img_path[14:-4]+"_mask.png"