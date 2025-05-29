import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiplicativeLR
from tqdm import tqdm

import wandb
from training_utils import ResidualMLP, parity_plot, checkpointing, run_epoch, get_dataloaders

if __name__ == '__main__':
    device = 'cuda'
    train_dataset_path = 'D:\crystal_datasets\ellipsoid_chunks\combined_ellipsoid_dataset.pt'
    test_dataset_path = 'ellipsoid_data_chunk0.pt'
    batch_size = 1000
    lr = 1e-3
    lr_lambda = 0.99
    hidden_dim = 512
    num_layers = 8
    dropout = 0
    norm = None
    max_epochs = 10000
    report_every = 5

    tr, te = get_dataloaders(batch_size, train_dataset_path, test_dataset_path)
    input_size = tr.dataset.x.shape[1]

    model = ResidualMLP(input_size, hidden_dim, 3, num_layers,
                        norm, dropout)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_lambda)

    with wandb.init(project='Ellipsoid_Overlap'):
        model.to(device)
        for epoch in tqdm(range(max_epochs)):
            train_loss, test_loss, target, output, loss = run_epoch(
                model, optimizer, tr, te, device, scheduler, epoch
            )
            if epoch % 5 == 0:
                wandb.log({'train_loss': torch.cat(train_loss).mean(),
                           'test_loss': torch.cat(test_loss).mean(),
                           'train_ov_error': torch.cat(train_loss)[:, 2].mean(),
                           'test_ov_error': torch.cat(test_loss)[:, 2].mean(),
                           'train_vol_error': torch.cat(train_loss)[:, :2].mean(),
                           'test_vol_error': torch.cat(test_loss)[:, :2].mean(),
                           'learning_rate': optimizer.param_groups[0]['lr'],
                           })

            if epoch % 10 == 0 and epoch > 10:
                checkpointing(model, test_loss)

            if (epoch % report_every == 0) and (epoch != 0):
                fig, corrcoef = parity_plot(target, output, loss)
                wandb.log({"Parity Plot": fig,
                           'R value': corrcoef})

    done = True
