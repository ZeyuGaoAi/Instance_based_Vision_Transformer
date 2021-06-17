from . import models


def build_model(args):
        
    if args.model == 'VitsCNN_pos':
        model = models.ViT.VitsCNN_pos(num_nuclei = args.num_nuclei,hidden_dim=args.hidden_dim,num_layers=args.num_layers,num_heads =args.num_heads,out_dim = args.num_classes,nuclues_size = args.nuclues_size)
      
    return model