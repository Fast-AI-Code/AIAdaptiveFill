from fastai.vision import *
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fastai.callbacks import *
from fastai.vision.gan import *
import cv2


def craptest(img):
    # targ_sz = resize_to(img, 96, use_min=True)
    # img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    # w, h = img.size
    img = img.filter(ImageFilter.FIND_EDGES)
    return img


learn = load_learner('gen-new')
learn.load('gen-pre3')


def image_run(img_name):
    temps = craptest(Image.open(f'/media/subhaditya/DATA/COSMO/PLAYGROUND/outline2Img/test/{img_name}.jpg')
                     ).save('/media/subhaditya/DATA/COSMO/PLAYGROUND/outline2Img/test/rem.jpg')
    ends = learn.predict(open_image('test/rem.jpg'))[0].save('test/pred.jpg')


image_run('temp')
