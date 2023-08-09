import argparse
from longformer_sequence_tagging_util import json_to_sequence_tagging_dataframe
from roberta_classification_util import *
from torch.utils.data import DataLoader
from eval import eval_across_domains
from load import load_prediction_and_gold


def evaluate(pred_path, gold_path, out_path):
    pred, gold = load_prediction_and_gold(pred_path, gold_path)

    per_domain, mean = eval_across_domains(gold, pred)

    with open(out_path, "w+") as f:
        for k, v in per_domain.items():
            f.write(f"f1_{k}:{v}\n")
        f.write(f"f1_mean:{mean}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/LOW_fold_0/train.csv", type=str, required=False)
    parser.add_argument("--validation", default="../data/LOW_fold_0/predictions_sim.csv", type=str, required=False)
    parser.add_argument("--text_path", default="../data/all", type=str, required=False)
    parser.add_argument("--test_json", type=str, default="../data/test_inputs_final.json", required=False)
    parser.add_argument("--model", type=str, default="roberta-large", required=False)
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--output", type=str, default="roberta_output", required=False)
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--max_norm", type=int, default=10, required=False)
    parser.add_argument("--extra_feature", type=str, default='', required=False)
    parser.add_argument("--plus_text", type=bool, default=False, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    extra_feature = False
    extra_dim = 0

    train = pd.read_csv(args.train)
    train['text'] = train['sentence']
    train['essay_id'] = train['id']
    validation = pd.read_csv(args.validation)
    validation_json = get_json_from_dataframe(validation, args.validation.replace('predictions_sim.csv','validation.json'))
    validation['text'] = validation['sentence']
    validation['essay_id'] = validation['id_x']
    for col in Constants.OUTPUT_LABELS:
        train[col] = np.where(train['label'] == col, 1, 0)
        validation[col] = np.where(validation['label'] == col, 1, 0)

    train_extra = []
    validation_extra = []

    tokenizer, model = build_model_tokenizer(args.model, extra_feature, extra_dim)

    train_dataset = PragTagDataset(train, max_len=args.max_len, tokenizer=tokenizer, data_path=args.text_path,
                                   plus_text=args.plus_text, extra_feature=train_extra)
    validation_dataset = PragTagDataset(validation, max_len=args.max_len, tokenizer=tokenizer, data_path=args.text_path,
                                        plus_text=args.plus_text, extra_feature=train_extra)
    train_data_loader = DataLoader(train_dataset,
                                   shuffle=True,
                                   batch_size=args.batch_size)

    val_data_loader = DataLoader(validation_dataset,
                                 shuffle=False,
                                 batch_size=args.batch_size)

    
    test_extra = []
    test = json_to_sequence_tagging_dataframe(pd.read_json(args.test_json), True)

    test_dataset = PragTagDatasetWithoutTargets(test, max_len=args.max_len, tokenizer=tokenizer, data_path=args.text_path, plus_text=args.plus_text, extra_feature=test_extra)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size=args.batch_size)
    
    
    test_preds, val_preds, model = train_model(n_epochs=args.epochs, train_loader=train_data_loader, val_loader=val_data_loader, test_loader=test_loader,
                        model=model, lr=args.lr, device=device)
    for epoch in range(args.epochs):
        output_dir = args.output+'/'+str(epoch)
        os.makedirs(output_dir, exist_ok=True)

        print(val_preds.get(epoch))
        val_output = get_final_prediction(validation_json, val_preds.get(epoch))
        val_output.to_json(output_dir+'/'+'validate_predicted.json', orient="records")
        evaluate(output_dir+'/'+'validate_predicted.json', args.validation.replace('predictions_sim.csv','validation.json'), output_dir+'/'+'scores.txt')

        test_output = get_final_prediction(pd.read_json(args.test_json), test_preds.get(epoch))
        test_output.to_json(output_dir +'/'+'predicted.json', orient="records")
    
