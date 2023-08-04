from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
from models import *
from metric import evaluate_auc
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
class LP_GRN:
    def __init__(self, args, eval_metric, device,
                 num_nodes, num_data):
        self.args = args
        self.device = device

        self.num_data = num_data
        self.use_feature = args.use_feature
        self.use_edge_weight = args.use_edge_weight
        self.eval_metric = eval_metric

        if args.train_node_embedding:
            self.emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
            torch.nn.init.xavier_uniform_(self.emb.weight)
        else:
            self.emb = None

        self.loss = BCEWithLogitsLoss()
        self.loss_rec = MSELoss()


        if args.model == 'DGCNN':
            self.model = DGCNN(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                               max_z=args.max_z, use_ignn=args.use_ignn, use_gatv2=args.use_gatv2,
                               num_heads=args.num_heads, k=args.k, num_features=args.num_features,
                               use_feature=args.use_feature, node_embedding=self.emb).to(device)
        if args.model == 'DGCNN_feat':
            self.model = DGCNN_feat(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                                    max_z=args.max_z, use_ignn=args.use_ignn, use_gatv2=args.use_gatv2,
                                    num_heads=args.num_heads, k=args.k, num_features=args.num_features,
                                    feature_dim=args.feature_dim, node_embedding=self.emb).to(device)

        if args.model == 'DGCNN_feat_noNeigFeat':
            self.model = DGCNN_feat_noNeigFeat(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                                    max_z=args.max_z, use_ignn=args.use_ignn, use_gatv2=args.use_gatv2,
                                    k=args.k, num_features=args.num_features,
                                    node_embedding=self.emb).to(device)

        if args.model == 'DGCNN_feat_rec':
            self.model = DGCNN_feat_rec(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                                        max_z=args.max_z, use_ignn=args.use_ignn, use_gatv2=args.use_gatv2,
                                        k=args.k, num_features=args.num_features,
                                        node_embedding=self.emb).to(device)
        elif args.model == 'SAGE':
            self.model = SAGE(hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                              max_z=args.max_z, use_feature=args.use_feature, node_embedding=self.emb).to(device)
        elif args.model == 'GCN':
            self.model = GCN(hidden_channels=args.hidden_channels, num_layers=args.num_layers, train_dataset=None,
                             max_z=args.max_z, node_embedding=self.emb).to(device)
        elif args.model == 'GIN':
            self.model = GIN(hidden_channels=args.hidden_channels, num_layers=args.num_layers, train_dataset=None,
                             max_z=args.max_z, node_embedding=self.emb).to(device)

        if args.pre_trained:
            ckpt = torch.load('DGCNN_pre_trained.pth')
            ckpt.pop('node_embedding.weight')
            if args.use_feature:
                ckpt.pop('convs.0.bias')
                ckpt.pop('convs.0.lin.weight')

                self.model.load_state_dict(ckpt, strict=False)
                param_train = ['convs.0.bias', 'convs.0.lin.weight',
                               'feat.0.weight', 'feat.0.bias', 'node_embedding.weight']
                self.optimizer = torch.optim.Adam([{'params': list(self.model.convs[0].parameters()) +
                                                         list(self.model.feat.parameters()) +
                                                         list(self.model.node_embedding.parameters()), 'lr': args.lr},
                                                   {'params': [p for n, p in list(self.model.named_parameters())
                                                         if not any(nd in n for nd in param_train)], 'lr': 0.1*args.lr}
                                                  ])
            else:
                self.model.load_state_dict(ckpt, strict=False)
                self.optimizer = torch.optim.Adam([{'params': self.model.node_embedding.parameters(), 'lr': args.lr},
                                                   {'params': [p for n, p in list(self.model.named_parameters())
                                                               if 'node' not in n], 'lr': 0.1*args.lr}
                                                  ])
        else:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.n_epochs, eta_min=1e-6)

        print(self.model)

    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        for data in tqdm(train_loader, ncols=70):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x if self.use_feature else None
            edge_weight = data.edge_weight if self.use_edge_weight else None
            node_id = data.node_id if self.emb else None
            if self.args.model == 'DGCNN_feat_rec':
                logits, x_rec = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
                loss = self.loss(logits.view(-1), data.y.to(torch.float)) + 0.01*self.loss_rec(x, x_rec)
            else:
                logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
                loss = self.loss(logits.view(-1), data.y.to(torch.float))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs

        self.scheduler.step()

        return total_loss / self.num_data


    @torch.no_grad()
    def val(self, val_loader):
        self.model.eval()

        y_pred, y_true = [], []
        total_loss = 0
        for data in tqdm(val_loader, ncols=70):
            data = data.to(self.device)
            x = data.x if self.use_feature else None
            edge_weight = data.edge_weight if self.use_edge_weight else None
            node_id = data.node_id if self.emb else None
            if self.args.model == 'DGCNN_feat_rec':
                logits, x_rec = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            else:
                logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))
            total_loss += self.loss(logits.view(-1), data.y.to(torch.float)).item() * data.num_graphs

        val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
        auc = evaluate_auc(val_pred, val_true)

        return total_loss / len(val_loader.dataset), auc

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()

        y_pred, y_true = [], []
        for data in tqdm(test_loader, ncols=70):
            data = data.to(self.device)
            x = data.x if self.use_feature else None
            edge_weight = data.edge_weight if self.use_edge_weight else None
            node_id = data.node_id if self.emb else None
            if self.args.model == 'DGCNN_feat_rec':
                logits, x_rec = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            else:
                logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))
        test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)

        auc = evaluate_auc(test_pred, test_true)

        return auc

    # @torch.no_grad()
    # def test_multiple_models(self, val_loader, models):
    #     for m in models:
    #         m.eval()
    #
    #     y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    #     for data in tqdm(val_loader, ncols=70):
    #         data = data.to(self.device)
    #         x = data.x if args.use_feature else None
    #         edge_weight = data.edge_weight if args.use_edge_weight else None
    #         node_id = data.node_id if emb else None
    #         for i, m in enumerate(models):
    #             logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
    #             y_pred[i].append(logits.view(-1).cpu())
    #             y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    #     val_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    #     val_true = [torch.cat(y_true[i]) for i in range(len(models))]
    #     pos_val_pred = [val_pred[i][val_true[i] == 1] for i in range(len(models))]
    #     neg_val_pred = [val_pred[i][val_true[i] == 0] for i in range(len(models))]
    #
    #     y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    #     for data in tqdm(test_loader, ncols=70):
    #         data = data.to(device)
    #         x = data.x if args.use_feature else None
    #         edge_weight = data.edge_weight if args.use_edge_weight else None
    #         node_id = data.node_id if emb else None
    #         for i, m in enumerate(models):
    #             logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
    #             y_pred[i].append(logits.view(-1).cpu())
    #             y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    #     test_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    #     test_true = [torch.cat(y_true[i]) for i in range(len(models))]
    #     pos_test_pred = [test_pred[i][test_true[i] == 1] for i in range(len(models))]
    #     neg_test_pred = [test_pred[i][test_true[i] == 0] for i in range(len(models))]
    #
    #     Results = []
    #     for i in range(len(models)):
    #         if args.eval_metric == 'hits':
    #             Results.append(evaluate_hits(pos_val_pred[i], neg_val_pred[i],
    #                                          pos_test_pred[i], neg_test_pred[i]))
    #         elif args.eval_metric == 'mrr':
    #             Results.append(evaluate_mrr(pos_val_pred[i], neg_val_pred[i],
    #                                         pos_test_pred[i], neg_test_pred[i]))
    #         elif args.eval_metric == 'rocauc':
    #             Results.append(evaluate_ogb_rocauc(pos_val_pred[i], neg_val_pred[i],
    #                                                pos_test_pred[i], neg_test_pred[i]))
    #
    #         elif args.eval_metric == 'auc':
    #             Results.append(evaluate_auc(val_pred[i], val_true[i],
    #                                         test_pred[i], test_pred[i]))
    #     return Results
    #

