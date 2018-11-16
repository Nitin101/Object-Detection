from PIL import Image
import os, sys

path = "flickr_logos_27/"
dirs = os.listdir( path )

def resize():
    c=0
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            if im.mode != "RGB":
                print("yes")
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)
            #
            # c += 1
            # print(c)


resize()