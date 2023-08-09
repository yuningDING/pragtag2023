import argparse
import pandas as pd
from longformer_sequence_tagging_util import *
from eval import eval_across_domains
from load import load_prediction_and_gold
from roberta_classification_util import get_terminology_replaced, KEEP_LIST

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
    parser.add_argument("--validation_json", default="../data/validation.json", type=str, required=False)
    parser.add_argument("--text_path", default="../data/all", type=str, required=False)
    parser.add_argument("--test_json", type=str, default="../data/test_inputs_final.json", required=False)
    parser.add_argument("--model", type=str, default="allenai/longformer-base-4096", required=False)
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--output", type=str, default="longformer_output", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--max_norm", type=int, default=10, required=False)
    parser.add_argument("--word_normalization", type=bool, default=False, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    # Step 1. Get args, seed everything and choose device
    args = parse_args()
    
    seed_everything(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    train = json_to_sequence_tagging_dataframe(pd.read_json(args.train_json))
    validation = json_to_sequence_tagging_dataframe(pd.read_json(args.validation_json))
    test = json_to_sequence_tagging_dataframe(pd.read_json(args.test_json), True)
    if args.word_normalization:
        train = get_terminology_replaced(train)
        validation = get_terminology_replaced(validation)
        test = get_terminology_replaced(test)

    train_preprocessed, train_gold = preprocess(train, args.text_path)
    validate_preprocessed, validate_gold = preprocess(validation, args.text_path)

    model, tokenizer = build_model_tokenizer(args.model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    training_set = ReviewDataset(train_preprocessed, tokenizer, args.max_len)
    training_loader = load_data(training_set, args.batch_size)

    for epoch in range(args.epochs):
        os.makedirs(args.output+'/'+str(epoch), exist_ok=True)
        print(f"Training epoch: {epoch + 1}")
        model_train(training_loader, model, optimizer, device, args.max_norm)
        print(f"Validating epoch: {epoch + 1}")
        validate_pred = model_predict(device, model, args.max_len, tokenizer, validate_preprocessed)
        validate_output = get_final_prediction(pd.read_json(args.validation_json), validate_pred)
        validate_output.to_json(args.output+'/'+str(epoch)+'/'+'validate_predicted.json', orient="records")
        evaluate(args.output+'/'+str(epoch)+'/'+'validate_predicted.json', args.validation_json ,args.output+'/'+str(epoch)+'/'+'scores.txt')

        test_preprocessed, test_gold = preprocess(test, args.text_path, True)
        test_pred = model_predict(device, model, args.max_len, tokenizer, test_preprocessed)
        test_output = get_final_prediction(pd.read_json(args.test_json), test_pred)
        test_output.to_json(args.output+'/'+str(epoch)+'/'+'predicted.json', orient="records")
