def get_norm_values(norm_family = 'imagenet'):
    '''
        Inputs
            norm_family: String of norm_family
        Returns
            mean, std : tuple of 3 channel values
    '''
    if norm_family == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError('Incorrect normalization family')
    return mean, std