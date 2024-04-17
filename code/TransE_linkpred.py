import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from pathlib import Path
from utils import set_seed, negative_sampling, print_args, set_config_args, remove_all_edges_of_etype
from data_processing import load_data4transE
from model import TransE, HeteroRGCN, HeteroLinkPredictionModel #, MarginRankingLoss
import copy


parser = argparse.ArgumentParser(description='Train a GNN-based link prediction model with TransE-encoder')
parser.add_argument('--device_id', type=int, default=-1)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='MIN_groundtruth')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=68)
parser.add_argument('--hidden_dim', type=int, default=68)
parser.add_argument('--out_dim', type=int, default=68)

'''
TransE args
'''
parser.add_argument('--margin', type=int, default=1.0)
parser.add_argument('--negative_rate', type=int, default=5, help='How many negative samples for per positive sample')
parser.add_argument('--alpha', type=int, default=0.5, help='TransE loss weight')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='drug', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='disease', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='treats', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')
parser.add_argument('--lr', type=float, default=0.01, help='link predictor learning_rate') 
parser.add_argument('--num_epochs', type=int, default=200, help='How many epochs to train')
parser.add_argument('--eval_interval', type=int, default=1, help="Evaluate once per how many epochs")
parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save the model')
parser.add_argument('--saved_model_dir', type=str, default='code/saved_models', help='Where to save the model')
parser.add_argument('--sample_neg_edges', default=True, action='store_true', 
                    help='If False, use fixed negative edges. If True, sample negative edges in each epoch')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

args = parser.parse_args()

if 'MIN' in args.dataset_name:
    args.src_ntype = 'drug'
    args.tgt_ntype = 'disease'
    args.pred_etype = 'treats'
elif 'synthetic' in args.dataset_name:
    args.src_ntype = 'user'
    args.tgt_ntype = 'item'
elif 'citation' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'paper'
else:
    raise ValueError('Unknow dataset argument')
    
if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'train_eval')
    
print_args(args)

def generate_negative_triples(g, cano_type, triples, negative_rate=1):
    src_ntype, edge_type, tgt_ntype = cano_type
    num_heads = g.num_nodes(ntype=src_ntype)
    num_relations = g.num_edges(etype=edge_type)
    num_tails = g.num_nodes(ntype=tgt_ntype)
    
    heads, relations, tails = triples[:, 0], triples[:, 1], triples[:, 2]
    
    neg_heads = torch.randint(num_heads, size=(len(triples) * negative_rate,))[:len(triples)]
    neg_tails = torch.randint(num_tails, size=(len(triples) * negative_rate,))[:len(triples)]
    if num_relations == 0:
        neg_relations = relations
    else:
        neg_relations = torch.randint(num_relations, size=(len(triples) * negative_rate,))[:len(triples)]
    
    # neg_triples = torch.stack([heads, relations, neg_tails]).t().contiguous()
    neg_triples = torch.stack([neg_heads, neg_relations, tails] if np.random.random() < 0.5
                              else [heads, neg_relations, neg_tails]).t().contiguous()  
    return neg_triples

def transE_loss(pos_score, neg_score):
    pos_scores = torch.cat([pos_score, neg_score])
    neg_scores = torch.cat([neg_score, pos_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.full((neg_score.shape[0],), -1)])
    return criterion(pos_scores, neg_scores, labels)

def LP_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    device = scores.device
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def run():
    set_seed(0)
    best_val_auc = 0
    best_val_loss = float('inf')
    best_epoch = None
    state = None # best model state
    # LP dataset
    pred_etype= args.pred_etype
    train_pos_src_nids, train_pos_tgt_nids = train_pos_g.edges(etype=pred_etype) 
    train_neg_src_nids, train_neg_tgt_nids = train_neg_g.edges(etype=pred_etype)           
    val_pos_src_nids, val_pos_tgt_nids = val_pos_g.edges(etype=pred_etype)            
    val_neg_src_nids, val_neg_tgt_nids = val_neg_g.edges(etype=pred_etype)            
    test_pos_src_nids, test_pos_tgt_nids = test_pos_g.edges(etype=pred_etype)            
    test_neg_src_nids, test_neg_tgt_nids = test_neg_g.edges(etype=pred_etype) 

    for epoch in range(1, args.num_epochs+1):
        # model.train()
        # comput transE loss
        train_pos_distances = torch.tensor([])
        train_neg_distances = torch.tensor([])
        mp_train_pos_g = remove_all_edges_of_etype(train_pos_g, pred_etype)
        for cano_type in mp_train_pos_g.canonical_etypes:
            src_ntype, edge_type, tgt_ntype = cano_type
            if edge_type == pred_etype:
                next
            else:
                train_pos_heads, train_pos_tails = mp_train_pos_g.edges(etype=edge_type)
                train_pos_relations = torch.arange(train_pos_heads.shape[0])
                train_pos_triples = torch.cat((train_pos_heads.view(1,-1), train_pos_relations.view(1,-1), train_pos_tails.view(1,-1)), dim=0).t()
                train_pos_distances = torch.cat((train_pos_distances, trans(cano_type, train_pos_triples)))
                
                train_neg_triples = generate_negative_triples(mp_train_pos_g, cano_type, train_pos_triples, args.negative_rate)
                train_neg_distances = torch.cat((train_neg_distances, trans(cano_type, train_neg_triples)))

        loss_transE = transE_loss(train_pos_distances, train_neg_distances)

        # compute LP loss
        train_pos_score = model(train_pos_src_nids, train_pos_tgt_nids, mp_g)   
        if args.sample_neg_edges:
            train_neg_src_nids, train_neg_tgt_nids = negative_sampling(train_pos_g, pred_etype) 
        train_neg_score = model(train_neg_src_nids, train_neg_tgt_nids, mp_g)
        loss_LP = LP_loss(train_pos_score, train_neg_score)

        total_loss = args.alpha*loss_transE + (1-args.alpha)*loss_LP
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                train_auc = compute_auc(train_pos_score, train_neg_score)[0]
                val_pos_score = model(val_pos_src_nids, val_pos_tgt_nids, mp_g)
                val_neg_score = model(val_neg_src_nids, val_neg_tgt_nids, mp_g)
                val_auc = compute_auc(val_pos_score, val_neg_score)[0]
                print('In epoch {}, loss: {:.4f}, train AUC: {:.4f}, val AUC: {:.4f}'.format(epoch, total_loss, train_auc, val_auc))
                if val_auc > best_val_auc:
                    best_epoch = epoch
                    best_val_auc = val_auc
                    state = copy.deepcopy(model.state_dict())
    
    with torch.no_grad():
        model.eval()
        model.load_state_dict(state)
        test_pos_score = model(test_pos_src_nids, test_pos_tgt_nids, mp_g)
        test_neg_score = model(test_neg_src_nids, test_neg_tgt_nids, mp_g)
        test_auc, test_auPRC = compute_auc(test_pos_score, test_neg_score)        
        print('Best epoch {}, val AUC: {:.4f}, test AUC: {:.4f}, test auPRC: {:.4f}'.format(best_epoch, best_val_auc, test_auc, test_auPRC))
    return test_auc
        

processed_g = load_data4transE(args.dataset_dir, args.dataset_name, args.pred_etype, args.valid_ratio, args.test_ratio)[1]
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in processed_g]

trans = TransE(mp_g, args.emb_dim)
encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim, trans)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MarginRankingLoss(margin=args.margin)

auc_on_test = run()


if args.save_model:
    output_dir = Path.cwd().joinpath(args.saved_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), output_dir.joinpath(f"TransE_{args.dataset_name}_model.pth"))

    print('--- saving model successfully. ---')




