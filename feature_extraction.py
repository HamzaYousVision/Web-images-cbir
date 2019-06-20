'''
Feature extraction from pretrained CNN for image similarity evaluation.

TODO:
1. load torch and torchvision libraries (PyTorch)
2. Load pretrained model from torchvision
    * Here we can test the use of different models (VGG, resnet, ...)
3. Extract features from the the activations.
    Here we have too many options, in:

    Model type:
    Many of the existing technique are based on classification models like VGG network. At the moment we can go for that.
    But I think that it worth trying other type of models: detection or segmentation.

    In order to have better visibility, try to:
        1. Visualize CNN activation from the intermediate layers and see the pattern. That's simple load some images and
        do the forward pass it through the model and have a look on the resulting pattern.

    Features level:
        1. From fc layer: in this case we lose the local information
        2. From convolution layers: the extraction could be from different lavel :
            2.1. Last convolution layers: this means that we are interested in the high level features (more complex)
            2.2. First or middle layers: this means that we are more intesrsted in the low level features (simple
            features)
            2.2. Multi-level features combination: low, mid and the high level by concatenation or more sophisticated
            fashion.

    Extraction technique:
        1. Global Pooling, (Max, Average)
        2. Feature eggregation Bag-of-Word, Vlad or ...
        3. Use the silency-map for better results.
4. Once we have the features extracted, we can use the description to find the image class in the dateset.
    * Here, we can use distance metrics like (euclidian, cosine, ...)

'''

# Load libraries
import torch #standard pytorch
from torch.autograd import Variable
import torchvision.models as models #import models from torchvision libraries
import numpy as np


# Load the models, that's so easy. Just like this, you can load an other models (VGG for instance)
model = models.resnet50(pretrained=True)
# Here, I recommand to try to run the inference (prediction for an image). You can use the function, preprocessing and
# make prediction in this script. This is just to show how the prediction is done.

# Extract features from the CNN model. For that i have added the function extract_layer to help you. (It is just an
# example that work for resnet50). We need more work on top of that to extract our feature.


def preprocess_image(im, resize_im=True):
    if resize_im:
        img = np.array(im.resize((224, 224)))
        im_arr = np.float32(img)/255
        im_arr = np.ascontiguousarray(im_arr[..., ::-1])
        im_arr = im_arr.transpose(2, 0, 1)
        im_ten = torch.from_numpy(im_arr).float()
        im_ten.unsqueeze_(0)
        im_var = Variable(im_ten, requires_grad=True)
        return im_var


def make_prediction(model, input_img):
    model.eval()
    out = model(input_img)

    prob, pred = torch.max(out, 1)
    pred = int(pred)
    return pred, prob


def extract_layer(model, l):
    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    if l == 1:
        model.layer1[2].relu.register_forward_hook(hook)
        weight = model.layer1[2].conv3.weight
    elif l == 2:
        model.layer2[3].relu.register_forward_hook(hook)
        weight = model.layer2[3].conv3.weight
    elif l == 3:
        model.layer3[5].relu.register_forward_hook(hook)
        weight = model.layer3[5].conv3.weight
    elif l == 4:
        model.layer4[2].relu.register_forward_hook(hook)
        weight = model.layer4[2].conv3.weight
    out = model(t_img)

    activ = outputs[0]

    mask_activ = get_mask(activ)
    mask_weight = get_mask(weight)

    return mask_weight, mask_activ
