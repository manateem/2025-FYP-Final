# This code was provided as being "code from older students" meaning none of the work here is our group's (until you reach the bottom where it is indicated where our work begins).
# There may be some adjustments made in order for this old code to work, but mostly not our own.

import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log, graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull

def get_compactness(mask, erosion_level):
    """Measures compactness of a mask. erosion_level controls how much is eroded when attempting to measure the perimeter."""
    mask = color.rgb2gray(mask)
    A = np.sum(mask)

    struct_el = morphology.disk(erosion_level)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = mask - mask_eroded
    l = np.sum(perimeter)

    compactness = (4*pi*A)/(l**2)

    score = round(1-compactness, 3)

    return {"compactness":compactness, "score":score}

def get_multicolor_rate(im, mask, n):
    mask = color.rgb2gray(mask)
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    mask = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
    )
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask[i][j] != 0:
                col_list.append(im2[i][j] * 256)

    if len(col_list) == 0:
        return ""

    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)

    if m <= 1:
        return ""

    for i in range(0, m - 1):
        j = i + 1
        col_1 = com_col_list[i]
        col_2 = com_col_list[j]
        dist_list.append(
            np.sqrt(
                (col_1[0] - col_2[0]) ** 2
                + (col_1[1] - col_2[1]) ** 2
                + (col_1[2] - col_2[2]) ** 2
            )
        )
    return np.max(dist_list)

def get_com_col(cluster, centroids):
    com_col_list = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    start = 0
    for percent, color in colors:
        if percent > 0.08:
            com_col_list.append(color)
        end = start + (percent * 300)
        cv2.rectangle(
            rect,
            (int(start), 0),
            (int(end), 50),
            color.astype("uint8").tolist(),
            -1,
        )
        start = end
    return com_col_list

def cut_mask(mask):
    
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum.all() != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum.all() != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def find_midpoint_v1(image):
    
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def asymmetry(mask):
    row_mid, col_mid = find_midpoint_v1(mask)

    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n: int):

    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    scores = asymmetry_scores.values()
    averaged_score = sum(scores)/len(scores)

    return {"scores": asymmetry_scores,"average":averaged_score}

def mean_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score          

def best_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    best_score = min(asymmetry_scores.values())

    return best_score

def worst_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    worst_score = max(asymmetry_scores.values())

    return worst_score  

def convexity_score(mask):

    coords = np.transpose(np.nonzero(mask))

    hull = ConvexHull(coords)

    lesion_area = np.count_nonzero(mask)

    convex_hull_area = hull.volume + hull.area

    convexity = lesion_area / convex_hull_area
    
    return convexity 

"""Functions Written By Manateem Below"""

def get_color_uniformity(img, mask):
    """
    Calculates the average color (scalar) and standard deviation (as a measure of variance)
    of a masked image region. The score is the average of standard deviations across RGB channels.
    """
    # Apply mask
    masked_pixels = img[mask > 0]

    if masked_pixels.size == 0:
        return {"score": 0, "average_color": 0}

    # Compute mean color per channel, then average the channels to get a single scalar
    average_color = np.mean(masked_pixels)  # Scalar: mean of all R, G, B values

    # Compute standard deviation per channel, then average
    std_dev = np.std(masked_pixels, axis=0)
    score = np.mean(std_dev)

    return {"score": score, "average_color": average_color}

def significant_color_count(img, mask, significance=0.05):
    """DOES NOT WORK AS INTENDED CURRENTLY, WIP"""
    
    """Uses kmeans to procedurally increase the number of colors until a color becomes 'insignificant', in which the number of colors before this point is returned."""
    """Assumes img that has already been masked, where non-lesion pixels are 0."""
    """Significance dictates level at which below a color would be considered insignificant."""

    safe_limit = 20 # just in case any unfortunate loop shenanigans occur

    mask = color.rgb2gray(mask)
    # (thresh, im_bw) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # pixel_count = np.sum(im_bw)

    pixel_count = np.sum(mask)

    success_count = 1

    for n in range(safe_limit):
        n_colors = n+1
        pixels = np.float32(img.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        for count in counts:
            if count/pixel_count < significance:
                return success_count
        success_count += 1
            
    return success_count
    
def convexity_metrics(mask):
    try:
        mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(mask_gray, 127, 255,0)
        contours,hierarchy = cv2.findContours(thresh,2,1)
        cnt = contours[0]
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        distances = []
        for i in range(defects.shape[0]):
            _,_,_,d = defects[i,0]
            distances.append(d)
    except:
        return {"variance":0, "average":0, "max":0, "score":0}
    avg = sum(distances)/len(distances)

    var = 0
    for i in range(len(distances)):
        var += (avg - i)**2

    return {"variance":var, "average":avg, "max":max(distances), "score":0}

def texture_analysis(img, mask):
    try:
        gray_img = rgb2gray(img)  # Now gray_img is 2D with float values in [0, 1]

        mask = mask[:, :, 0]
        gray_img = (gray_img * 255).astype(np.uint8)

        # Apply the mask
        masked_img = gray_img.copy()
        masked_img[mask == 0] = 0
        glcm = graycomatrix(masked_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    except:
        return None


    return {"glcm_contrast": contrast, "glcm_energy":energy, "glcm_homogeneity":homogeneity}

def color_metrics(img, mask):
    def insertionSort(arr):
        n = len(arr)
        if n <= 1:
            return 
        for i in range(1, n):
            key = arr[i]
            j = i-1
            while j >= 0 and key < arr[j]:
                arr[j+1] = arr[j]
                j -= 1
            arr[j+1] = key

    masked_img = img.copy()
    masked_img[mask==0] = 0
    masked_gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)

    return {"max_brightness": masked_gray.max(), "min_brightness": masked_gray[masked_gray != 0].min()}
def get_avg_max_redness(img, mask, percentile=99.9):
    """
    Calculates the average of the top 0.1% red channel values within a masked region.
    
    Args:
        img (np.ndarray): RGB image (H x W x 3).
        mask (np.ndarray): Binary or boolean mask (H x W), non-zero means active pixel.
        percentile (float): The percentile threshold (default is 99.9 for top 0.1%).
        
    Returns:
        float: Average of top red values within the mask.
    """
    # Extract red channel values where the mask is active
    red_channel = img[:, :, 0]
    masked_red = red_channel[mask > 0]

    if masked_red.size == 0:
        return 0.0  # Handle case where mask is empty

    # Determine threshold for top 0.1%
    threshold = np.percentile(masked_red, percentile)

    # Select values above threshold
    top_red_values = masked_red[masked_red >= threshold]

    # Compute and return average
    return float(np.mean(top_red_values))
