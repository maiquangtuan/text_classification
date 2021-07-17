from models import *
from train import *
import torch
import argparse
from torch.utils.data import DataLoader, random_split
from torch import optim
from data import *
from torchtext.datasets import AG_NEWS
train_iter = AG_NEWS(split='train')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train text classification model')
parser.add_argument('--embedding_dim', type = int, default = 100)
parser.add_argument('--hidden_dim', type = int, default = 100)
parser.add_argument('--num_class', type = int, default = 4)
parser.add_argument('--n_layers', type = int, default = 2)
parser.add_argument('--bidirectional', type = bool, default = False)
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--ratio', type=float, default=0.85)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--num_epochs', type=int, default = 10)
parser.add_argument('--save_dir', type = str)

def main():
    args = parser.parse_args()
    train_iter, test_iter = AG_NEWS()
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])


    dataloaders = {
	'train': DataLoader(train_iter, batch_size= args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory = True,prefetch_factor=1, collate_fn=collate_batch),
	'valid': DataLoader(test_iter, batch_size= args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=1, collate_fn=collate_batch)
    }

    model = LSTM_model(vocab_size= len(vocab), embedding_dim= args.embedding_dim, hidden_dim= args.hidden_dim,
    num_class=args.num_class, n_layers= args.n_layers, bidirectional=args.bidirectional, dropout= 0.5)
    trloss_val, tsloss_val = [], []   #list of loss function value

    optimz = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.num_epochs + 1):
        print('Epoch:', epoch)
        train_iter(model, optimz, dataloaders['train'], trloss_val, args.device)
        evaluate(model, dataloaders['valid'], tsloss_val)
 
    
    torch.save(model.state_dict(), f'{args.save_dir}/model_epoch_{args.num_epochs}.pt')

if __name__ == '__main__':
    main()