import pandas as pd
from longformer_sequence_tagging_util import json_to_sequence_tagging_dataframe
from longformer_sequence_tagging_pipeline import evaluate
from roberta_classification_pipeline import parse_args
from roberta_classification_util import *
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder


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
    validation = json_to_sequence_tagging_dataframe(pd.read_json(args.validation_json))
    test = json_to_sequence_tagging_dataframe(pd.read_json(args.test_json), True)

    for col in Constants.OUTPUT_LABELS:
        train[col] = np.where(train['label'] == col, 1, 0)
        validation[col] = np.where(validation['label'] == col, 1, 0)

    #train = train.head()
    #validation = validation.head()
    #test = test.head()

    ohe = OneHotEncoder()
    train_extra = []
    validation_extra = []
    test_extra = []

    if args.extra_feature == 'domain':
        extra_dim = 5
        extra_feature = True
        train_extra = ohe.fit_transform(train[['domain']]).toarray()
        validation_extra = ohe.fit_transform(validation[['domain']]).toarray()
        test_extra = ohe.fit_transform(test[['domain']]).toarray()

    tokenizer, model = build_model_tokenizer(args.model, extra_feature, extra_dim)
    
    train_dataset = PragTagDataset(train, max_len=args.max_len, tokenizer=tokenizer, data_path=args.text_path+'/train', plus_text=args.plus_text, extra_feature=train_extra)
    
    validation_dataset = PragTagDataset(validation, max_len=args.max_len, tokenizer=tokenizer, data_path=args.text_path+'/train',
                                        plus_text=args.plus_text, extra_feature=validation_extra)
    train_data_loader = DataLoader(train_dataset,
                                   shuffle=True,
                                   batch_size=args.batch_size)

    val_data_loader = DataLoader(validation_dataset,
                                 shuffle=False,
                                 batch_size=args.batch_size)

    test_dataset = PragTagDatasetWithoutTargets(test, max_len=args.max_len, tokenizer=tokenizer,
                                                data_path=args.text_path+'/test', plus_text=args.plus_text, extra_feature=test_extra)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    test_preds, val_preds, model = train_model(n_epochs=args.epochs, train_loader=train_data_loader,
                                               val_loader=val_data_loader, test_loader=test_loader,
                                               model=model, lr=args.lr, device=device)
                                               #, extra_data=args.extra_feature)
    for epoch in range(args.epochs):
        output_dir = args.output + '/' + str(epoch)
        os.makedirs(output_dir, exist_ok=True)

        val_output = get_final_prediction(pd.read_json(args.validation_json), val_preds.get(epoch))
        val_output.to_json(output_dir + '/' + 'validate_predicted.json', orient="records")
        evaluate(output_dir + '/' + 'validate_predicted.json', args.validation_json, output_dir + '/' + 'scores.txt')

        test_output = get_final_prediction(pd.read_json(args.test_json), test_preds.get(epoch))
        test_output.to_json(output_dir + '/' + 'predicted.json', orient="records")

