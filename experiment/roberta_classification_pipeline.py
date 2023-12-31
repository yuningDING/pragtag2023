import argparse
from longformer_sequence_tagging_util import json_to_sequence_tagging_dataframe
from roberta_classification_util import *
from torch.utils.data import DataLoader
from eval import eval_across_domains
from load import load_prediction_and_gold
from sklearn.preprocessing import OneHotEncoder


def evaluate(pred_path, gold_path, out_path):
    pred, gold = load_prediction_and_gold(pred_path, gold_path)

    per_domain, mean = eval_across_domains(gold, pred)

    with open(out_path, "w+") as f:
        for k, v in per_domain.items():
            f.write(f"f1_{k}:{v}\n")
        f.write(f"f1_mean:{mean}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="../data/train.json", type=str, required=False)
    parser.add_argument("--with_auxiliary", type=bool, default=False, required=False)
    parser.add_argument("--auxiliary_data", default="../data/arr.json", type=str, required=False)
    parser.add_argument("--validation_json", default="../data/validation.json", type=str, required=False)
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
    parser.add_argument("--word_normalization", type=bool, default=False, required=False)
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

    train = json_to_sequence_tagging_dataframe(pd.read_json(args.train_json))
    if args.with_auxiliary is True:
        if 'json' in args.auxiliary_data:
            aux = json_to_sequence_tagging_dataframe(pd.read_json(args.auxiliary_data))
        else:
            aux = pd.read_csv(args.auxiliary_data)
        train = pd.concat([train, aux], ignore_index=True)
        print('Using auxiliary data. Training size: ' + str(train.shape))
    validation = json_to_sequence_tagging_dataframe(pd.read_json(args.validation_json))
    test = json_to_sequence_tagging_dataframe(pd.read_json(args.test_json), True)


    for col in Constants.OUTPUT_LABELS:
        train[col] = np.where(train['label'] == col, 1, 0)
        validation[col] = np.where(validation['label'] == col, 1, 0)

    if args.word_normalization:
        train = replace_terminology(train)
        validation = replace_terminology(validation)
        test = replace_terminology(test, True)

    domains = ["case", "diso", "iscb", "rpkg", "scip", "secret"]
    ohe = OneHotEncoder(categories=[domains])
    train_extra = []
    validation_extra = []
    test_extra = []

    if args.extra_feature == 'domain':
        extra_dim = 6
        extra_feature = True
        train_extra = ohe.fit_transform(train[['domain']]).toarray()
        validation_extra = ohe.fit_transform(validation[['domain']]).toarray()
        test_extra = ohe.fit_transform(test[['domain']]).toarray()
    elif args.extra_feature == 'position':
        extra_dim = 2
        extra_feature = True
        train_extra = get_position(train)
        validation_extra = get_position(validation)
        test_extra = get_position(test)
    elif args.extra_feature == 'both':
        extra_dim = 8
        extra_feature = True
        train_domain = ohe.fit_transform(train[['domain']]).toarray()
        validation_domain = ohe.fit_transform(validation[['domain']]).toarray()
        test_domain = ohe.fit_transform(test[['domain']]).toarray()
        train_position = get_position(train)
        validation_position = get_position(validation)
        test_position = get_position(test)
        for i in range(len(train_position)):
            train_extra.append([*train_domain[i], *train_position[i]])
            i += 1
        for i in range(len(validation_position)):
            validation_extra.append([*validation_domain[i], *validation_position[i]])
            i += 1
        for i in range(len(test_position)):
            test_extra.append([*test_domain[i], *test_position[i]])
            i += 1

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

    test_dataset = PragTagDatasetWithoutTargets(test, max_len=args.max_len, tokenizer=tokenizer, data_path=args.text_path, plus_text=args.plus_text, extra_feature=test_extra)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size=args.batch_size)
    
    
    test_preds, val_preds, model = train_model(n_epochs=args.epochs, train_loader=train_data_loader, val_loader=val_data_loader, test_loader=test_loader,
                        model=model, lr=args.lr, device=device)
    for epoch in range(args.epochs):
        output_dir = args.output+'/'+str(epoch)
        os.makedirs(output_dir, exist_ok=True)

        print(val_preds.get(epoch))
        val_output = get_final_prediction(pd.read_json(args.validation_json), val_preds.get(epoch))
        val_output.to_json(output_dir+'/'+'validate_predicted.json', orient="records")
        evaluate(output_dir+'/'+'validate_predicted.json', args.validation_json, output_dir+'/'+'scores.txt')

        test_output = get_final_prediction(pd.read_json(args.test_json), test_preds.get(epoch))
        test_output.to_json(output_dir +'/'+'predicted.json', orient="records")
    
