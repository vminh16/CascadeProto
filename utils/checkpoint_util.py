""" Util functions for loading and saving checkpoints
"""
import os
import torch

# Save the original torch.load function
_original_torch_load = torch.load

# Define a new function that forces weights_only=False
def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

# Override torch.load globally
torch.load = custom_torch_load


def load_pretrain_checkpoint(model, pretrain_checkpoint_path):
    # load pretrained model for point cloud encoding
    model_dict = model.state_dict()
    if pretrain_checkpoint_path is not None:
        print('Load encoder module from pretrained checkpoint...')
        pretrained_dict = torch.load(os.path.join(pretrain_checkpoint_path, 'checkpoint.tar'))['params']
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise ValueError('Pretrained checkpoint must be given.')

    return model

def save_pretrain_checkpoint(model, output_path):
    torch.save(dict(params=model.encoder.state_dict()), os.path.join(output_path, 'checkpoint.tar'))


def load_model_checkpoint(model_checkpoint_path):
    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.pt'))
        iter = checkpoint['iteration']
        iou = checkpoint['IoU']
        print('Load model checkpoint at Iteration %d (IoU %f)...' % (iter, iou))
        return checkpoint['model']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)
