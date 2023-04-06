from packaging import version
import pytorch_lightning as pl

def fix_gpu_args(args):
    # Different ways of asking for GPUs
    if version.parse(pl.__version__) >= version.parse("1.7.0") and args.gpus:
        args.accelerator='gpu'
        args.devices=args.gpus
        args.gpus=None

# handle version details
def get_trainer_kwargs(args, find_unused_parameters=False):
    use_spawn = False
    trainer_kwargs = dict()
    # DistributedDataParallel
    if version.parse(pl.__version__) >= version.parse("1.6.0"):
        from pytorch_lightning.strategies.ddp import DDPStrategy
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=find_unused_parameters)
    elif version.parse(pl.__version__) >= version.parse("1.5.0"):
        from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin
        trainer_kwargs["strategy"] = (DDPSpawnPlugin(find_unused_parameters=find_unused_parameters) if use_spawn # causes dataloader issues
            else DDPPlugin(find_unused_parameters=find_unused_parameters))
    else:
        from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin
        trainer_kwargs["plugins"] = (DDPSpawnPlugin(find_unused_parameters=find_unused_parameters) if use_spawn # causes dataloader issues
            else DDPPlugin(find_unused_parameters=find_unused_parameters))

    return trainer_kwargs