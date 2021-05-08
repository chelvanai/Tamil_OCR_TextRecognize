from PIL import Image, ImageDraw, ImageFont
from tamil.txt2unicode import unicode2bamini

width = 250
height = 60
back_ground_color = (255, 255, 255)
font_color = (0, 0, 0)
unicode_font = ImageFont.truetype("/home/chelvan/.local/share/fonts/Unknown Vendor/TrueType/Bamini/Bamini_Regular.ttf",
                                  28)

with open('corpus.bcn.test.ta') as f:
    l = f.readlines()

a = 1
for i in l:
    for j in i[:-1].split(" "):
        print(j)
        try:
            im = Image.new("RGB", (width, height), back_ground_color)
            draw = ImageDraw.Draw(im)
            tscii = unicode2bamini(j)
            draw.text((5, 5), tscii, font=unicode_font, fill=font_color)
            im.save("./images2/" + str(a) + "_" + str(j) + "_.jpg")
        except Exception as e:
            print(e)
        a += 1
