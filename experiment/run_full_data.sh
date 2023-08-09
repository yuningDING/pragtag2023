python3.8 roberta_classification_pipeline.py --output roberta_large_baseline --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_plus_word_normalization --word_normalization True --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_plus_Pos --extra_feature position --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_plus_Domain --extra_feature domain --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_plus_Text --plus_text True --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_plus_Pos_Text --extra_feature position --plus_text True --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_plus_Pos_Text_Domain --extra_feature both --plus_text True --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_with_ARR --with_auxiliary True --epochs 10
python3.8 roberta_classification_pipeline.py --output roberta_large_with_ARR_sampled --with_auxiliary True --auxiliary_data "../data/arr_sampled.csv" --epochs 10

