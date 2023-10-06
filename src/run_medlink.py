from beir.datasets.data_loader import GenericDataLoader

from src.dataset.utils import get_train_dataloader, get_eval_dataloader
from src.helper import Helper
from src.model.backbone import Backbone
from src.utils import *


def parse_arguments(parser):
    parser.add_argument('--dataset', type=str, default='mimic3')
    parser.add_argument('--code', type=str, default='CCS_CODE')
    parser.add_argument('--model', type=str, default='medlink')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--neg_sample', type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--monitor", type=str, default="Recall@1")
    parser.add_argument("--monitor_criterion", type=str, default="max")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--no-train', type=bool, default=False)
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--exp-name-attr', type=list, default=['dataset', 'code', 'model', 'note'])
    parser.add_argument("--official-run", action="store_true", default=False)
    parser.add_argument("--no-cuda", type=bool, default=False)
    args = parser.parse_args()
    return args


helper = Helper(parse_arguments)
args = helper.args

""" load data """
read_path = os.path.join(data_path, f'{args.dataset}/processed/{args.dataset}_{args.code}')
tr_spt = 'train_w_neg' if args.neg_sample else 'train'
train_corpus, train_queries, train_qrels = GenericDataLoader(read_path).load(split=tr_spt)
val_corpus, val_queries, val_qrels = GenericDataLoader(read_path).load(split='val')
test_corpus, test_queries, test_qrels = GenericDataLoader(read_path).load(split='test')
candidate = read_json(os.path.join(data_path, f'{args.dataset}/processed/candidate.json'))

train_dataloader = get_train_dataloader(
    train_corpus, train_queries, train_qrels, batch_size=args.batch_size, shuffle=True
)
val_corpus_dataloader, val_queries_dataloader = get_eval_dataloader(
    val_corpus, val_queries, batch_size=args.batch_size
)
test_corpus_dataloader, test_queries_dataloader = get_eval_dataloader(
    test_corpus, test_queries, batch_size=args.batch_size
)

""" load model """
model = Backbone(args)
model.to(args.device)
logging.info(model)
logging.info("Number of parameters: {}".format(count_parameters(model)))

""" train """
if args.checkpoint:
    helper.load_checkpoint(model, args.checkpoint)

if not args.no_train:
    for epoch in range(args.epochs):

        logging.info("-------train: {}-------".format(epoch))
        scores = model.train_epoch(train_dataloader)
        for key in scores:
            helper.log(f"metrics/train/{key}", scores[key])
        helper.save_checkpoint(model, "last.ckpt")

        logging.info("-------val: {}-------".format(epoch))
        scores = model.eval_epoch(val_corpus_dataloader, val_queries_dataloader, val_qrels, candidate,
                                  bootstrap=False)
        for key in scores.keys():
            helper.log(f"metrics/val/{key}", scores[key])
        helper.save_checkpoint_if_best(model, "best.ckpt", scores)

        logging.info("-------test: {}-------".format(epoch))
        scores = model.eval_epoch(test_corpus_dataloader, test_queries_dataloader, test_qrels, candidate,
                                  bootstrap=False)
        for key in scores.keys():
            helper.log(f"metrics/test/{key}", scores[key])

        if not args.official_run:
            break

    helper.load_checkpoint(model, os.path.join(helper.model_saved_path, "best.ckpt"))

""" final test """
logging.info("-------final test-------")
scores = model.eval_epoch(test_corpus_dataloader, test_queries_dataloader, test_qrels, candidate, bootstrap=True)
for key in scores.keys():
    helper.log(f"metrics/final_test/{key}", scores[key])
