from PIL import Image
import torchvision.transforms as transforms

def read_image(image_name):
    # change type of image to Tensor
    loader = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()])
    
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) # convert image dim to 3 dimension
    transforms.ToTensor()
    
    print(f"{image.size()} size of image is loaded..")
    return image

