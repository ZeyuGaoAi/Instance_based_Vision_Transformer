import torch.optim as optim


def build_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            args.lr,
            #betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            #eps=args.eps,
        )
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            args.lr,
            alpha=args.rms_alpha,
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        raise ValueError(
            "Invalid optimizer specified: {}".format(args.optimizer)
        )

    return optimizer
