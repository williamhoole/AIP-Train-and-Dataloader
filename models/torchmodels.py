import torch
import torchvision.models as models

def get_model(model_name, num_classes, pretrained=True):
    # Dictionary mapping model names to their respective functions in torchvision.models
    model_funcs = {
        'vgg16': models.vgg16,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'alexnet': models.alexnet,
        'squeezenet': models.squeezenet1_0,
        'densenet': models.densenet161,
        'inception': models.inception_v3,
        'swin_t': models.swin_t,
        'swin_b': models.swin_b,


        # Add more models as needed
    }

    # Check if the model name is valid
    if model_name not in model_funcs:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(model_funcs.keys())}")

    # Load the model
    model = model_funcs[model_name](pretrained=pretrained)

    # Modify the last layer to have the correct number of classes
    if 'resnet' in model_name:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'vgg'or 'alexnet' in model_name:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif 'squeezenet' in model_name:
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    elif 'densenet' in model_name:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif 'inception' in model_name:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'swin' in model_name:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not yet supported for custom number of classes.")

    return model