def get_opt(name):
    if name == 'cvpr':
        from config.opt_cvpr import opt
    elif name == 'rbdd':
        from config.opt_rbdd import opt
    elif name == 'mit':
        from config.opt_mit import opt
    return opt
