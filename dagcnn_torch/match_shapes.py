from torch.nn import functional

from .largest_dimensions import largest_dimensions

def match_shapes(tensors, match_channels=True):
    (max_feature_depth, max_height, max_width) = largest_dimensions(tensors)
    result = []
    for tensor in tensors:
        if match_channels:
            channel_difference = max_feature_depth - tensor.size(1)
        else:
            channel_difference = 0
        channel_padding_front = channel_difference // 2
        channel_padding_back = channel_difference - channel_padding_front
        
        height_difference = max_height - tensor.size(2)
        height_padding_top = height_difference // 2
        height_padding_bottom = height_difference - height_padding_top
        
        width_difference = max_width - tensor.size(3)
        width_padding_left = width_difference // 2
        width_padding_right = width_difference - width_padding_left

        padding = (width_padding_left, width_padding_right, height_padding_top, height_padding_bottom, channel_padding_front, channel_padding_back)
        padded_tensor = functional.pad(tensor, padding)
        result.append(padded_tensor)
    return result
