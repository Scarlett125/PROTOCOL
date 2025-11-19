import os
from easydict import EasyDict

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_config(args):

    cfg = EasyDict()
    cfg.update(args.__dict__)
    output_dir = args.output_dir
    mkdir_if_missing(output_dir)

    cfg['dataset']=args.dataset
    if cfg['setup'] in ['cluster']:
        if "cluster_dir" in args.__dict__ and args.cluster_dir is not None:
            cluster_dir = args.cluster_dir
        else:
            cluster_dir = os.path.join(output_dir, 'cluster')
        mkdir_if_missing(cluster_dir)
        cfg['cluster_dir'] = cluster_dir
        cfg['cluster_checkpoint'] = os.path.join(cluster_dir, 'checkpoint.pth.tar')
        cfg['cluster_model'] = os.path.join(cluster_dir, 'model.pth.tar')

    return cfg
