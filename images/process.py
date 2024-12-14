from PIL import Image

image_path = ['predict_Enet'+str(i)+'.png' for i in range(1,5)]
print(image_path)

for image in image_path:
    with Image.open(image) as im:
        cropbox = (70,140,590,330)
        cropped_image = im.crop(cropbox)
        cropped_image.save("cropped_" +image)
        #cropped_image.show()
