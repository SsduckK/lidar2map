import cv2
import numpy as np


def show_imgs(titled_imgs, wnd_title, columns, min_width=None, wait_key=0):
    # convert gray images to bgr images
    titled_imgs = gray_to_bgr(titled_imgs)
    # prepare merged image
    rows = int(np.ceil(len(titled_imgs)/columns))
    block_size, img_rect = get_block_info(titled_imgs)
    result_img = np.ones((int(block_size[0]*rows), int(block_size[1]*columns), 3), dtype=np.uint8)*255
    # put images on blocks with title on top of image
    for idx, (title, image) in enumerate(titled_imgs.items()):
        y_pos = int(idx / columns) * block_size[0] + img_rect["y"]
        x_pos = int(idx % columns) * block_size[1] + img_rect["x"]
        result_img[y_pos:y_pos+img_rect["h"], x_pos:x_pos+img_rect["w"], :] = image
        title_pos = (x_pos + 20, y_pos - 5)
        cv2.putText(result_img, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    # pad image with white
    if min_width is not None and result_img.shape[1] < min_width:
        result_img = pad_img(result_img, min_width)
    cv2.imshow(wnd_title, result_img)
    cv2.waitKey(wait_key)
    return result_img


def gray_to_bgr(images):
    bgr_imgs = dict()
    for key, img in images.items():
        if len(img.shape) == 2:
            bgr_imgs[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr_imgs[key] = img.copy()
    return bgr_imgs


def get_block_info(imgs_dict):
    pad = 10
    title_pad = 25
    first_key = list(imgs_dict.keys())[0]
    img_size = imgs_dict[first_key].shape
    img_area = {"x": pad, "w": img_size[1], "y": title_pad, "h": img_size[0]}
    block_size = (img_area["h"] + title_pad + pad, img_area["w"] + pad*2)
    return block_size, img_area


def pad_img(srcimg, dst_width):
    im_width = srcimg.shape[1]
    dstimg = np.ones((srcimg.shape[0], dst_width, 3), dtype=np.uint8) * 255
    start_col = int((dst_width - im_width)/2)
    end_col = start_col + im_width
    dstimg[:, start_col:end_col, :] = srcimg
    return dstimg
