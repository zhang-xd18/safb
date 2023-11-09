import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import SVDataLoader
from datetime import datetime
import os


def init_dir():
    dir_name = '{}-cr{}-L{}'.format(args.mode, args.cr, args.L)
    if args.debug:
        save_path = os.path.join(args.root, 'debug', dir_name)
    else:
        save_path = os.path.join(args.root, 'release', dir_name + '-{0:%m%d-%H%M}'.format(datetime.now()))
    if args.evaluate:
        save_path = os.path.join(args.root, 'evaluate', dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.set_file(os.path.join(save_path + '/log.txt'))
    return save_path


def main():
    save_path = init_dir()
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    logger.info('{} - {} - cr: {} ---- L: {} '.format(args.mode, 'debug' if args.debug else 'release', args.cr, args.L))

    # Environment initialization
    device = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    train_loader, val_loader, test_loader = SVDataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        # L=6,
        L=args.L,
        device=device)()

    # Define model
    model = init_model(args)
    model.to(device)

    # Define loss function
    test_criterion = nn.MSELoss().to(device)
    train_criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(args.mode, model, device, test_criterion)(test_data=test_loader)
        return

    # Define optimizer and scheduler
    lr_init = 1e-3 if args.scheduler == 'const' else 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)

    # Define the training pipeline
    trainer = Trainer(mode=args.mode,
                      model=model,
                      device=device,
                      optimizer=optimizer,
                      train_criterion=train_criterion,
                      test_criterion=test_criterion,
                      scheduler=scheduler,
                      save_path=save_path,
                      val_freq=1,
                      test_freq=1)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader, save_trend=True)

    # Final testing
    loss, nmse = Tester(args.mode, model, device, test_criterion)(test_data=test_loader)
          
    logger.info(f"\n=! Final test loss: {loss:.3e}"
                f"\n         test NMSE: {nmse:.3e}")


if __name__ == "__main__":
    main()
