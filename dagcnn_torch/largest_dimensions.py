def largest_dimensions(tensors):
    feature_depths = map(lambda tensor: tensor.size(1), tensors)
    heights = map(lambda tensor: tensor.size(2), tensors)
    widths = map(lambda tensor: tensor.size(3), tensors)
    max_feature_depth = max(feature_depths)
    max_height = max(heights)
    max_width = max(widths)
    return(max_feature_depth, max_height, max_width)
