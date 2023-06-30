import argparse
import pandas as pd
from longformer_sequence_tagging_util import *
from sklearn.model_selection import train_test_split
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
    parser.add_argument("--train_json", default="../data/train_inputs_full.json", type=str, required=False)
    parser.add_argument("--train_txt", default="../data/review_text/train", required=False)
    parser.add_argument("--test_json", type=str, default="../data/test_inputs.json", required=False)
    parser.add_argument("--test_txt", type=str, default="../data/review_text/test", required=False)
    parser.add_argument("--model", type=str, default="allenai/longformer-base-4096", required=False)
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--output", type=str, default="longformer_output_without_validation", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=9, required=False)
    parser.add_argument("--max_norm", type=int, default=10, required=False)
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

    input_data = pd.read_json(args.train_json)
    # train, validate = train_test_split(input_data, test_size=0.2)
    # validate.to_json('validate.json', orient="records")

    # train_preprocessed, train_gold = preprocess(train, args.train_txt)
    # validate_preprocessed, validate_gold = preprocess(validate, args.train_txt)

    train_preprocessed, train_gold = preprocess(input_data, args.train_txt)
    model, tokenizer = build_model_tokenizer(args.model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    training_set = ReviewDataset(train_preprocessed, tokenizer, args.max_len)
    training_loader = load_data(training_set, args.batch_size)

    os.makedirs(args.output, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"Training epoch: {epoch + 1}")
        model_train(training_loader, model, optimizer, device, args.max_norm)
        # print(f"Validating epoch: {epoch + 1}")
        # validate_pred = model_predict(device, model, args.max_len, tokenizer, validate_preprocessed)
        # validate_output = get_final_prediction(validate, validate_pred)
        # validate_output.to_json(args.output+'/'+'validate_predicted_'+str(epoch)+'.json', orient="records")
        # evaluate(args.output+'/'+'validate_predicted_'+str(epoch)+'.json', 'validate.json',args.output+'/'+'scores_'+str(epoch)+'.txt')

    test = pd.read_json(args.test_json)
    test_preprocessed, test_gold = preprocess(test, args.test_txt, True)
    test_pred = model_predict(device, model, args.max_len, tokenizer, test_preprocessed)
    test_output = get_final_prediction(test, test_pred)
    test_output.to_json(args.output+'/'+'predicted.json', orient="records")
